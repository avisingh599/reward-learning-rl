import argparse


def parse_args_avi(subparsers):
    avi_parser = subparsers.add_parser('avi', help='use if your name is avi')
    avi_parser.add_argument(
        '--image', type=str, help='docker image name', default='softlearning:20181212-dev-cpu-v1')
    avi_parser.add_argument(
        '--local-output-dir',
        type=str,
        help='output directory in local machine',
        default='/media/avi/data/Work/softlearning/data')
    avi_parser.add_argument(
        '--softlearning-dir',
        type=str,
        help='location of sac repo',
        default='/media/avi/data/Work/softlearning')
    # avi_parser.add_argument(
    #     '--gym-dir',
    #     type=str,
    #     help='location of gym-larry repo',
    #     default='/media/avi/data/Work/proj_4/gym-larry')
    # avi_parser.add_argument('-g', '--gpu', type=int, help='which gpu to use')

def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(title='user')
    parse_args_avi(subparsers)
    #parse_args_larry(subparsers)
    return parser.parse_args()
