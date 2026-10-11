#!/usr/bin/env python3
"""Measure whether ViTLSTM commands point away from four synthetic walls."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'models'))
from model import LSTMNetVIT  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--dataset', type=Path, default=Path('training/datasets/pure_walls'))
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    model = LSTMNetVIT().to(args.device).float()
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    all_margins = []
    with torch.no_grad():
        for side in ('left', 'right', 'ceiling', 'floor'):
            margins = []
            for trajectory in sorted(args.dataset.glob(f'wall_{side}_*')):
                meta = pd.read_csv(trajectory / 'data.csv')
                images = np.stack([
                    cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    for path in sorted(trajectory.glob('*.png'))
                ]).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(images[1:]).to(args.device).unsqueeze(1)
                desired = torch.from_numpy(meta.desired_vel.to_numpy(dtype=np.float32)[1:]).to(args.device).unsqueeze(1)
                quat = torch.from_numpy(meta[['quat_1', 'quat_2', 'quat_3', 'quat_4']].to_numpy(dtype=np.float32)[1:]).to(args.device)
                prediction, _ = model([image_tensor, desired, quat])
                component = 1 if side in ('left', 'right') else 2
                away_sign = -1.0 if side in ('left', 'ceiling') else 1.0
                margins.append((away_sign * prediction[:, component]).cpu().numpy())

            margins = np.concatenate(margins)
            all_margins.append(margins)
            print(f'{side:7s} away_mean={margins.mean():+.4f} '
                  f'min={margins.min():+.4f} toward={(margins < 0).mean() * 100:5.2f}%')

    margins = np.concatenate(all_margins)
    print(f'ALL     away_mean={margins.mean():+.4f} min={margins.min():+.4f} '
          f'toward={(margins < 0).mean() * 100:5.2f}% ({(margins < 0).sum()}/{len(margins)})')


if __name__ == '__main__':
    main()
