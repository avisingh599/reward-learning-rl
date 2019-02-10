"""
Example script for running a python program on a docker image.

Usage example:
    python scripts/docker_interactive.py avi
"""
from __future__ import print_function

import argparse
import os
from os.path import abspath, dirname, realpath

import doodad as dd
import doodad.mount as mount
from mount_args import parse_args


def main(args):
    mode = dd.mode.LocalDocker(image=args.image, gpu=True)

    mount_points = [
        # mount.MountLocal(
        #     local_dir=args.sac_dir,
        #     mount_point='/root/sac-plus',
        #     pythonpath=True),
        # mount.MountLocal(
        #     local_dir=args.gym_dir,
        #     mount_point='/root/gym-larry',
        #     pythonpath=True),
        # mount.MountLocal(
        #     local_dir=args.local_output_dir,
        #     mount_point='/mount/outputs',
        #     output=True),
        mount.MountLocal(
            local_dir=args.softlearning_dir,
            mount_point='/root/softlearning',
            pythonpath=True),
        # mount.MountLocal(
        #     local_dir=args.gym_dir,
        #     mount_point='/root/gym-larry',
        #     pythonpath=True),
        mount.MountLocal(
            local_dir=args.local_output_dir,
            mount_point='/root/ray_results',
            output=True),
    ]

    command = 'cd /root/softlearning'
    #command += '; python -c \"import mujoco_py\"'
    # if args.gpu is not None:
    #     command += '; export CUDA_VISIBLE_DEVICES={}'.format(args.gpu)

    dd.launch_interactive(
        mode=mode, command=command, mount_points=mount_points, verbose=False)


if __name__ == '__main__':
    main(parse_args())
