import jittor as jt
import argparse


def convent_checkpoints(checkpoint_path):
    model = jt.load(checkpoint_path)
    jt.save(model['state_dict'], checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default='')
    args = parser.parse_args()
    convent_checkpoints(args.checkpoint_path)

