from __future__ import print_function

import argparse
import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.multiprocessing as mp
from torch.optim import Adam

import agent.my_optim as my_optim
from agent.envs import create_atari_env
from agent.model import ActorCritic, ActorCritic_Substitude
from agent.train import train
from attack.rl_vaegan_defense import transfer_defense


# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-episode-length', type=int, default=1000000, help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='BreakoutDeterministic-v4', help='environment to train on (default: BreakoutDeterministic-v4 | PongDeterministic)')

parser.add_argument('--test-attacker', default='fgsm', help='adversary attack algorithms: fgsm|rand_fgsm|cw2')
parser.add_argument('--test-epsilon-adv', type=float, default=0.003, help='epsilon perturbation for attack model.')

parser.add_argument('--which-epoch', type=str, default='00380000', help='using specific epoch trained rl_vaegan model to defense')

parser.add_argument('--gpu-id', type=int, default=0, help='epsilon perturbation for attack model.')

if __name__ == '__main__':
    mp.set_start_method("spawn")
    os.environ['OMP_NUM_THREADS'] = '1'

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)


    torch.manual_seed(args.seed)
    env = create_atari_env(args.env_name, args)
    if args.black_box_attack:
        shared_model = ActorCritic_Substitude(env.observation_space.shape[0], env.action_space)
    else:
        shared_model = ActorCritic(env.observation_space.shape[0], env.action_space)
    shared_model.share_memory()

    if args.no_shared:
        optimizer = None
    else:
        optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
        optimizer.share_memory()

    # load a pre-trained model according to the ft-setting

    if args.ft_setting == 'full-ft':
        if args.env_name == 'BreakoutDeterministic-v4':
            fname = './agent/trained_model/breakout/11000.pth.tar'
        elif args.env_name == 'PongDeterministic-v4':
            fname = './agent/trained_model/pong/4000.pth.tar'
        else:
            sys.exit('Only support Break or Pong')
        
        print(fname)
        if os.path.isfile(fname):
            checkpoint = torch.load(fname)
            shared_model.load_state_dict(checkpoint['state_dict'])
            for param in shared_model.parameters():
                param.requires_grad = True
            if 'partial' in args.ft_setting:
                for param in shared_model.conv1.parameters():
                    param.requires_grad = False
                for param in shared_model.conv2.parameters():
                    param.requires_grad = False
                for param in shared_model.conv3.parameters():
                    param.requires_grad = False

            if 'random' in args.ft_setting:
                shared_model.init_output_layers()

                if 'partial' in args.ft_setting:
                    shared_model.init_conv_lstm()

            print("model was loaded successfully")

    processes = []

    counter = mp.Value('i', 0)

    transfer_defense(args.num_processes, args, shared_model, counter)
