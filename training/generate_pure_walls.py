#!/usr/bin/env python3
"""Generate deterministic planar-wall depth trajectories for ViTFly.

The images use the simulator's depth convention: 0 is at the camera and 255
is the 9 m clipping distance. Each scene contains exactly one planar side wall
and open space everywhere else.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


HEADER = [
    '', 'timestamp', 'desired_vel', 'quat_1', 'quat_2', 'quat_3', 'quat_4',
    'pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'velcmd_x',
    'velcmd_y', 'velcmd_z', 'ct_cmd', 'br_cmd_x', 'br_cmd_y', 'br_cmd_z',
    'is_collide',
]


def wall_depth_image(side, distance, boundary, roll, rng, surface='parallel', clip_distance=9.0, height=60, width=90):
    """Render z-depth to a side plane through a pinhole camera."""
    fov_x = math.radians(70.0)
    fx = width / (2.0 * math.tan(fov_x / 2.0))
    fy = fx
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx - (width - 1) / 2.0
    y = yy - (height - 1) / 2.0
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    xr = cos_r * x + sin_r * y
    yr = -sin_r * x + cos_r * y

    depth = np.full((height, width), clip_distance, dtype=np.float32)
    if side == 'left':
        edge = (boundary - 0.5) * width
        mask = xr < edge
        depth[mask] = distance if surface == 'front' else distance * fx / np.maximum(edge - xr[mask], 0.5)
    elif side == 'right':
        edge = (boundary - 0.5) * width
        mask = xr > edge
        depth[mask] = distance if surface == 'front' else distance * fx / np.maximum(xr[mask] - edge, 0.5)
    elif side == 'ceiling':
        edge = (boundary - 0.5) * height
        mask = yr < edge
        depth[mask] = distance if surface == 'front' else distance * fy / np.maximum(edge - yr[mask], 0.5)
    elif side == 'floor':
        edge = (boundary - 0.5) * height
        mask = yr > edge
        depth[mask] = distance if surface == 'front' else distance * fy / np.maximum(yr[mask] - edge, 0.5)
    else:
        raise ValueError(side)

    # Match an 8-bit simulator recording without introducing scene texture.
    image = np.clip(depth / clip_distance * 255.0, 1.0, 255.0)
    image += rng.normal(0.0, 0.6, image.shape)
    return np.clip(np.rint(image), 1, 255).astype(np.uint8)


def generate(output, trajectories_per_side, frames, seed, clip_distance):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    desired_vel = 5.0
    sides = ('left', 'right', 'ceiling', 'floor')

    for side_index, side in enumerate(sides):
        for trajectory_index in range(trajectories_per_side):
            trajectory = output / f'wall_{side}_{trajectory_index:03d}'
            trajectory.mkdir(parents=True, exist_ok=True)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            surface = 'front' if trajectory_index % 2 == 0 else 'parallel'
            base_distance = rng.uniform(0.25, 1.1)
            base_boundary = rng.uniform(0.38, 0.62)
            base_roll = rng.uniform(-0.10, 0.10)
            rows = []

            for frame in range(frames):
                progress = frame / max(frames - 1, 1)
                distance = np.clip(
                    base_distance + 0.20 * math.sin(2.0 * math.pi * progress + phase),
                    0.15,
                    1.3,
                )
                boundary = np.clip(
                    base_boundary + 0.045 * math.sin(4.0 * math.pi * progress + phase),
                    0.32,
                    0.68,
                )
                roll = base_roll + 0.025 * math.sin(2.0 * math.pi * progress)
                image = wall_depth_image(
                    side, distance, boundary, roll, rng,
                    surface=surface, clip_distance=clip_distance,
                )
                timestamp = 1000.0 * (side_index + 1) + 10.0 * trajectory_index + 0.04 * frame
                cv2.imwrite(str(trajectory / f'{timestamp:.3f}.png'), image)

                # Stronger avoidance for closer walls and larger occupied areas.
                urgency = np.clip((2.6 - distance) / 2.35, 0.35, 1.0)
                away = 2.2 + 1.5 * urgency
                vel_y = -away if side == 'left' else away if side == 'right' else 0.0
                vel_z = -away if side == 'ceiling' else away if side == 'floor' else 0.0
                vel_x = math.sqrt(max(desired_vel ** 2 - away ** 2, 0.25))
                rows.append([
                    frame, f'{timestamp:.3f}', desired_vel, 1.0, 0.0, 0.0, 0.0,
                    frame * 0.04 * vel_x, 0.0, 3.5, vel_x, vel_y, vel_z,
                    vel_x, vel_y, vel_z, 10.0, 0.0, 0.0, 0.0, False,
                ])

            with (trajectory / 'data.csv').open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(HEADER)
                writer.writerows(rows)

    print(f'Generated {len(sides) * trajectories_per_side} trajectories and '
          f'{len(sides) * trajectories_per_side * frames} frames in {output}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('training/datasets/pure_walls'))
    parser.add_argument('--trajectories-per-side', type=int, default=12)
    parser.add_argument('--frames', type=int, default=80)
    parser.add_argument('--seed', type=int, default=20260716)
    parser.add_argument('--clip-distance', type=float, default=9.0)
    args = parser.parse_args()
    generate(args.output, args.trajectories_per_side, args.frames, args.seed, args.clip_distance)


if __name__ == '__main__':
    main()
