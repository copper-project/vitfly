"""
@authors: A Bhattacharya
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the training routine that was used in the paper "Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al
"""

import glob
import os, sys
import random
from os.path import join as opj
import numpy as np
import torch
from datetime import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

from dataloading import *
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models'))
import model as model_library

# NOTE this suppresses tensorflow warnings and info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import getpass
uname = getpass.getuser()

# a class that trains an network to predict actions from depth images
# trainer can be loaded in two ways:
# 1. just for dataloading, in which case dataset_name is provided and usually no_model=True, or
# 2. for model training, in which case just args is provided
class TRAINER:
    def __init__(self, args=None):
        self.args = args
        if self.args is not None:
            self.device = args.device
            self.basedir = args.basedir
            self.logdir = args.logdir
            self.datadir = args.datadir
            self.ws_suffix = args.ws_suffix
            self.dataset_name = args.dataset
            self.extra_dataset_name = args.extra_dataset
            self.extra_dataset_names = [name.strip() for name in args.extra_dataset.split(',') if name.strip()]
            repeat_values = [int(value.strip()) for value in str(args.extra_dataset_repeats).split(',')]
            if len(repeat_values) == 1:
                repeat_values *= len(self.extra_dataset_names)
            if len(repeat_values) != len(self.extra_dataset_names):
                raise ValueError('--extra_dataset_repeats must contain one value or one value per extra dataset')
            self.extra_dataset_repeats = repeat_values
            self.extra_horizontal_flip = args.extra_horizontal_flip
            self.short = args.short

            self.model_type = args.model_type
            self.val_split = args.val_split
            self.seed = args.seed # if args.seed>0 else None
            self.load_checkpoint = args.load_checkpoint
            self.checkpoint_path = args.checkpoint_path
            self.lr = args.lr
            self.N_eps = args.N_eps
            self.lr_warmup_epochs = args.lr_warmup_epochs
            self.lr_decay = args.lr_decay
            self.save_model_freq = args.save_model_freq
            self.val_freq = args.val_freq
            self.avoidance_loss_weight = args.avoidance_loss_weight
            self.toward_wall_loss_weight = args.toward_wall_loss_weight

            if self.seed is not None:
                random.seed(self.seed)
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)

        else:
            raise Exception("Args are not provided")


        assert self.dataset_name is not None, 'Dataset name not provided, neither through args nor through dataset_name kwarg'

        ###############
        ## Workspace ##
        ###############

        expname = datetime.now().strftime('d%m_%d_t%H_%M')
        self.workspace = opj(self.basedir, self.logdir, expname)
        wkspc_ctr = 2
        while os.path.exists(self.workspace):
            self.workspace = opj(self.basedir, self.logdir, expname+f'_{str(wkspc_ctr)}')
            wkspc_ctr += 1
        self.workspace = self.workspace + self.ws_suffix
        os.makedirs(self.workspace)
        self.writer = SummaryWriter(self.workspace)

        # save ordered args, config, and a logfile to write stdout to
        if self.args is not None:
            f = opj(self.workspace, 'args.txt')
            with open(f, 'w') as file:
                for arg in sorted(vars(self.args)):
                    attr = getattr(self.args, arg)
                    file.write('{} = {}\n'.format(arg, attr))
                f = opj(self.workspace, 'config.txt')
            with open(f, 'w') as file:
                file.write(open(self.args.config, 'r').read())
        f = opj(self.workspace, 'log.txt')
        self.logfile = open(f, 'w')

        self.mylogger(f'[LearnerLSTM init] Making workspace {self.workspace}')

        self.dataset_dir = opj(self.datadir, self.dataset_name)

        #################
        ## Dataloading ##
        #################

        if self.load_checkpoint:
            print('[LearnerLSTM init] Loading train_val_dirs from checkpoint')
            try:
                train_val_dirs = tuple(np.load(opj(os.path.dirname(self.checkpoint_path), 'train_val_dirs.npy'), allow_pickle=True))
            except:
                print('[LearnerLSTM init] Could not load train_val_dirs from checkpoint, dataloading from scratch')
                train_val_dirs = None
        else:    
            train_val_dirs = None

        if train_val_dirs is None and self.extra_dataset_name:
            train_val_dirs = self.mixed_train_val_dirs()

        self.dataloader(val_split=self.val_split, short=self.short, seed=self.seed, train_val_dirs=train_val_dirs)

        # TODO hardcoding num_training_steps to be the number of trajectories instead of number of images
        self.num_training_steps = self.train_trajlength.shape[0]
        self.num_val_steps = self.val_trajlength.shape[0]
        self.lr_warmup_iters = self.lr_warmup_epochs * self.num_training_steps
        extra_dataset_dirs = {
            os.path.abspath(opj(self.basedir, self.datadir, name))
            for name in self.extra_dataset_names
        }
        self.train_is_extra = np.array([
            any(os.path.commonpath((os.path.abspath(path), extra_dir)) == extra_dir for extra_dir in extra_dataset_dirs)
            for path in self.train_dirs
        ])

        ##################################
        ## Define network and optimizer ##
        ##################################

        self.mylogger('[SETUP] Establishing model and optimizer.')
        if self.model_type == 'LSTMNet':
            self.model = model_library.LSTMNet().to(self.device).float()
        elif self.model_type == 'ConvNet':
            self.model = model_library.ConvNet().to(self.device).float()
        elif self.model_type == 'ViT':            
            self.model = model_library.ViT().to(self.device).float()
        elif self.model_type == 'ViTLSTM':
            self.model = model_library.LSTMNetVIT().to(self.device).float()
        elif self.model_type == 'UNet':
            self.model = model_library.UNetConvLSTMNet().to(self.device).float()
        else:
            self.mylogger(f'[SETUP] Invalid model_type {self.model_type}. Exiting.')
            exit()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.num_eps_trained = 0
        if self.load_checkpoint:
            self.load_from_checkpoint(self.checkpoint_path)

        self.total_its = self.num_eps_trained * self.num_training_steps

    def mixed_train_val_dirs(self):
        rng = random.Random(self.seed)

        def split_dataset(path):
            folders = sorted(folder for folder in glob.glob(opj(path, '*')) if os.path.isdir(folder))
            if not folders:
                raise RuntimeError(f'No trajectories found in {path}')
            rng.shuffle(folders)
            num_val = int(self.val_split * len(folders))
            if self.val_split > 0 and len(folders) > 1:
                num_val = max(1, num_val)
            return folders[num_val:], folders[:num_val]

        base_train, base_val = split_dataset(opj(self.basedir, self.dataset_dir))
        train_dirs = list(base_train)
        val_dirs = list(base_val)
        extra_summary = []
        for name, repeats in zip(self.extra_dataset_names, self.extra_dataset_repeats):
            extra_dir = opj(self.basedir, self.datadir, name)
            extra_train, extra_val = split_dataset(extra_dir)
            train_dirs += extra_train * repeats
            val_dirs += extra_val
            extra_summary.append(f'{len(extra_train)} {name} x{repeats}')
        rng.shuffle(train_dirs)

        self.mylogger(
            f'[DATALOADER] Mixed split: {len(base_train)} base + '
            f'{" + ".join(extra_summary)} training trajectories; '
            f'{len(base_val)} base + {len(val_dirs) - len(base_val)} unique extra validation trajectories'
        )
        return train_dirs, val_dirs

    def mylogger(self, msg):
        print(msg)
        self.logfile.write(msg+'\n')

    def load_from_checkpoint(self, checkpoint_path):
        try:
            self.num_eps_trained = int(checkpoint_path[-10:-4])
        except:
            self.num_eps_trained = 0
            self.mylogger(f'[SETUP] Could not parse number of epochs trained from checkpoint path {checkpoint_path}, using 0')
        self.mylogger(f'[SETUP] Loading checkpoint from {checkpoint_path}, already trained for {self.num_eps_trained} epochs')
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    def dataloader(self, val_split, short=0, seed=None, train_val_dirs=None):
        self.mylogger(f'[DATALOADER] Loading from {self.dataset_dir}')
        train_data, val_data, is_png, (self.train_dirs, self.val_dirs) = dataloader(opj(self.basedir, self.dataset_dir), val_split=val_split, short=short, seed=seed, train_val_dirs=train_val_dirs)
        self.train_meta, self.train_ims, self.train_trajlength, self.train_desvel, self.train_currquat, self.train_currctbr = train_data
        self.val_meta, self.val_ims, self.val_trajlength, self.val_desvel, self.val_currquat, self.val_currctbr = val_data
        self.mylogger(f'[DATALOADER] Dataloading done | train images {self.train_ims.shape}, val images {self.val_ims.shape}')

        self.train_meta, self.train_ims, self.train_desvel, self.train_currquat, self.train_currctbr = preload((self.train_meta, self.train_ims, self.train_desvel, self.train_currquat, self.train_currctbr), self.device)
        self.val_meta, self.val_ims, self.val_desvel, self.val_currquat, self.val_currctbr = preload((self.val_meta, self.val_ims, self.val_desvel, self.val_currquat, self.val_currctbr), self.device)
        self.mylogger(f'[DATALOADER] Preloading into device {self.device} done')

        assert self.train_ims.max() <= 1.0 and self.train_ims.min() >= 0.0, 'Images not normalized (values outside [0.0, 1.0])'
        assert self.train_ims.max() > 0.50, "Images not normalized (values only below 0.10, possibly due to not normalizing images from 'old' dataset)"

        self.train_velcmd = self.train_meta[:, range(13, 16) if is_png else range(12, 15)]
        self.val_velcmd = self.val_meta[:, range(13, 16) if is_png else range(12, 15)]

        # save train and val dirs in workspace for later use
        np.save(opj(self.workspace, 'train_val_dirs.npy'), np.array((self.train_dirs, self.val_dirs), dtype=object))

    def lr_scheduler(self, it):
        if it < self.lr_warmup_iters:
            lr = (0.9*self.lr)/self.lr_warmup_iters * it + 0.1*self.lr
        else:
            if self.lr_decay:
                lr = self.lr * (0.1 ** ((it-self.lr_warmup_iters) / (self.N_eps*self.num_training_steps)))
            else:
                lr = self.lr
        return lr

    def save_model(self, ep):
        self.mylogger(f'[SAVE] Saving model at epoch {ep}')
        path = self.workspace
        torch.save(self.model.state_dict(), opj(path, f'model_{str(ep).zfill(6)}.pth'))
        self.mylogger(f'[SAVE] Model saved at {path}')

    def weighted_mse_loss(self, input, target, weight):
        return torch.mean(weight * (input - target) ** 2)

    def command_loss(self, prediction, target):
        if self.avoidance_loss_weight <= 0:
            return F.mse_loss(prediction, target)

        # Most frames command straight flight. Without this weighting their
        # forward-velocity error overwhelms each short avoidance maneuver.
        component_weight = torch.ones_like(target)
        avoidance_strength = torch.clamp(target[:, 1:].abs() / 0.15, max=1.0)
        component_weight[:, 1:] += self.avoidance_loss_weight * avoidance_strength
        loss = self.weighted_mse_loss(prediction, target, component_weight)

        # A wrong-sign lateral/vertical command is qualitatively worse than a
        # weak correct command: it steers into the obstacle. Penalize only
        # components with a meaningful avoidance label.
        if self.toward_wall_loss_weight > 0:
            avoidance_mask = target[:, 1:].abs() >= 0.15
            toward_component = torch.relu(-prediction[:, 1:] * torch.sign(target[:, 1:]))
            if avoidance_mask.any():
                loss += self.toward_wall_loss_weight * torch.mean(
                    toward_component[avoidance_mask] ** 2
                )
        return loss


    def train(self):

        self.mylogger(f'[TRAIN] Training for {self.N_eps} epochs')
        train_start = time.time()

        # starting indices of trajectories in dataset
        self.train_traj_starts = np.cumsum(self.train_trajlength) - self.train_trajlength
        train_traj_lengths = self.train_trajlength

        for ep in range(self.num_eps_trained, self.num_eps_trained + self.N_eps):

            # periodically save model checkpoint
            if ep % self.save_model_freq == 0 and ep - self.num_eps_trained > 0:
                self.save_model(ep)

            # periodically evaluate on validation set
            if ep % self.val_freq == 0:
                self.validation(ep)

            ep_loss = 0
            gradnorm = 0

            # shuffling order of training data trajectories here
            # since we index data using train_traj_starts, we can just shuffle that!
            shuffled_traj_indices = np.random.permutation(len(self.train_traj_starts))
            train_traj_starts = self.train_traj_starts[shuffled_traj_indices]
            train_traj_lengths = self.train_trajlength[shuffled_traj_indices]
            train_is_extra = self.train_is_extra[shuffled_traj_indices]

            ### Training loop ###
            self.model.train()
            for it in range(self.num_training_steps):
                self.optimizer.zero_grad()
                traj_input = self.train_ims[train_traj_starts[it]+1 : train_traj_starts[it]+train_traj_lengths[it], :, :].unsqueeze(1) #1, traj, 
                desvel = self.train_desvel[train_traj_starts[it]+1 : train_traj_starts[it]+train_traj_lengths[it]].view(-1, 1)
                currquat = self.train_currquat[train_traj_starts[it]+1 : train_traj_starts[it]+train_traj_lengths[it]]
                cmd = self.train_velcmd[train_traj_starts[it]+1 : train_traj_starts[it]+train_traj_lengths[it], :]
                cmd_norm = cmd / desvel # normalize each row by each desvel element

                if self.extra_horizontal_flip and train_is_extra[it] and np.random.random() < 0.5:
                    traj_input = torch.flip(traj_input, dims=(-1,))
                    currquat = currquat.clone()
                    currquat[:, (1, 3)] *= -1
                    cmd_norm = cmd_norm.clone()
                    cmd_norm[:, 1] *= -1

                pred, _ = self.model([traj_input, desvel, currquat]) #, (init_hidden_state, init_cell_state)])
                loss = self.command_loss(pred, cmd_norm)
                ep_loss += loss
                loss.backward()
                gradnorm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=torch.inf)
                self.optimizer.step()
                new_lr = self.lr_scheduler(self.total_its-self.num_eps_trained*self.num_training_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                self.total_its += 1

            ep_loss /= self.num_training_steps
            gradnorm /= self.num_training_steps

            self.mylogger(f'[TRAIN] Completed epoch {ep + 1}/{self.num_eps_trained + self.N_eps}, ep_loss = {ep_loss:.6f}, time = {time.time() - train_start:.2f}s, time/epoch = {(time.time() - train_start)/(ep + 1 - self.num_eps_trained):.2f}s')

            self.writer.add_scalar('train/loss', ep_loss, ep)
            self.writer.add_scalar('train/gradnorm', gradnorm, ep)
            self.writer.add_scalar('train/lr', new_lr, self.total_its)
            self.writer.flush()

        self.mylogger(f'[TRAIN] Training complete, total time = {time.time() - train_start:.2f}s')
        self.save_model(ep)

    def validation(self, ep):

        self.mylogger(f'[VAL] Validating for val set of size {self.val_ims.shape[0]} images')

        val_start = time.time()
        it = 1

        with torch.no_grad():

            ep_loss = 0

            # starting index of trajectories in dataset
            val_traj_starts = np.cumsum(self.val_trajlength) - self.val_trajlength
            val_traj_starts = np.hstack((val_traj_starts, -1)) # -1 as the end of 

            ### Validation loop ###
            self.model.eval()

            for it in range(self.num_val_steps):
                # init_hidden_state = torch.rand(self.model.lstm.num_layers, 1, 3).to(self.device).float()
                # init_hidden_state[0] = self.train_velcmd[val_traj_starts[it], :].view(1, 1, -1)
                # init_cell_state = torch.rand(self.model.lstm.num_layers, 1, 3).to(self.device).float()

                traj_input = self.val_ims[val_traj_starts[it]+1 : val_traj_starts[it]+self.val_trajlength[it], :, :].unsqueeze(1)
                desvel = self.val_desvel[val_traj_starts[it]+1 : val_traj_starts[it]+self.val_trajlength[it]].view(-1, 1)
                currquat = self.val_currquat[val_traj_starts[it]+1 : val_traj_starts[it]+self.val_trajlength[it]]
                pred, _ = self.model([traj_input, desvel, currquat]) #, (init_hidden_state, init_cell_state)])
                cmd = self.val_velcmd[val_traj_starts[it]+1 : val_traj_starts[it]+self.val_trajlength[it], :]
                cmd_norm = cmd / desvel # normalize each row by each desvel element
                loss = F.mse_loss(cmd_norm, pred)
                ep_loss += loss

            ep_loss /= (it+1)

            self.mylogger(f'[VAL] Completed validation, val_loss = {ep_loss:.6f}, time taken = {time.time() - val_start:.2f} s')
            self.writer.add_scalar('val/loss', ep_loss, ep)

def argparsing():

    import configargparse
    parser = configargparse.ArgumentParser()

    # general params
    parser.add_argument('--config', is_config_file=True, help='config file relative path')
    parser.add_argument('--basedir', type=str, default=f'/home/{uname}/agile_ws/src/agile_flight', help='path to repo')
    parser.add_argument('--logdir', type=str, default='learner/logs', help='path to relative logging directory')
    parser.add_argument('--datadir', type=str, default=f'/home/{uname}/agile_ws/src/agile_flight', help='path to relative dataset directory')
    
    # experiment-level and learner params
    parser.add_argument('--ws_suffix', type=str, default='', help='suffix if any to workspace name')
    parser.add_argument('--model_type', type=str, default='LSTMNet', help='string matching model name in lstmArch.py')
    parser.add_argument('--dataset', type=str, default='5-2', help='name of dataset')
    parser.add_argument('--extra_dataset', type=str, default='', help='additional dataset mixed into training only with optional repetition')
    parser.add_argument('--extra_dataset_repeats', type=str, default='1', help='one repeat count or comma-separated counts matching extra_dataset')
    parser.add_argument('--extra_horizontal_flip', action='store_true', default=False, help='randomly mirror extra-dataset trajectories and their lateral labels')
    parser.add_argument('--short', type=int, default=0, help='if nonzero, how many trajectory folders to load')
    parser.add_argument('--val_split', type=float, default=0.2, help='fraction of dataset to use for validation')
    parser.add_argument('--seed', type=int, default=None, help='random seed to use for python random, numpy, and torch -- WARNING, probably not fully implemented')
    parser.add_argument('--device', type=str, default='cuda', help='generic cuda device; specific GPU should be specified in CUDA_VISIBLE_DEVICES')
    parser.add_argument('--load_checkpoint', action='store_true', default=False, help='whether to load from a model checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=f'/home/{uname}/agile_ws/src/agile_flight/learner/logs/d05_10_t03_13/model_000499.pth', help='absolute path to model checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--N_eps', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr_warmup_epochs', type=int, default=5, help='number of epochs to warmup learning rate for')
    parser.add_argument('--lr_decay', action='store_true', default=False, help='whether to use lr_decay, hardcoded to exponentially decay to 0.01 * lr by end of training')
    parser.add_argument('--save_model_freq', type=int, default=25, help='frequency with which to save model checkpoints')
    parser.add_argument('--val_freq', type=int, default=10, help='frequency with which to evaluate on validation set')
    parser.add_argument('--avoidance_loss_weight', type=float, default=0.0, help='extra training weight for labeled lateral and vertical avoidance commands')
    parser.add_argument('--toward_wall_loss_weight', type=float, default=0.0, help='extra hinge penalty for avoidance predictions with the wrong sign')

    args = parser.parse_args()
    print(f'[CONFIGARGPARSE] Parsing args from config file {args.config}')

    return args

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    args = argparsing()
    print(args)

    learner = TRAINER(args)
    learner.train()
