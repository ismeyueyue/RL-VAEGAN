import time
from collections import deque

import torch
import torch.nn.functional as F
from breakout_a3c.envs import create_atari_env
from breakout_a3c.model import ActorCritic
from os import listdir
from os.path import isfile, join

import foolbox
from breakout_a3c.adversary import Attack
from breakout_a3c.generate_adversary import FGSM
def test(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    if args.test_gan:
        log_name = 'breakout_a3c/' + args.gan_dir
        gan_path = args.gan_models_path + args.gan_dir + '/checkpoints'
        # gan_path = '/home/huyueyue/CODE/GAN/RL_GAN_ItoI/unit/outputs/breakout-diagonals/checkpoints'
        files = [join(gan_path, f).split('_')[-1].split('.')[0] for f in listdir(gan_path) if isfile(join(gan_path, f)) and f.startswith('gen')]
        gan_file = files.pop(-1)
        env = create_atari_env(args.env_name, args, True, gan_file)
    else:
        env = create_atari_env(args.env_name, args)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model.load_state_dict(shared_model.state_dict())
    model.eval()

    state = env.reset()
    # state = torch.from_numpy(state)
    done = True
    reward_ep = []
    # model.load_state_dict(shared_model.state_dict())
    for episode in range(5):
    # while True:
    #     env.render()
        is_Terminal = False
        frame_length = 0
        # model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)
            hx = torch.zeros(1, 256)
        else:
            cx = cx.detach()
            hx = hx.detach()
        rewards = 0
        pre_hx, pre_cx = hx, cx
        # for step in range(args.num_steps):
        while not is_Terminal:
            frame_length += 1
            # env.render()
            state = torch.from_numpy(state)
            value, logit, (hx, cx) = model((state.unsqueeze(0), (pre_hx, pre_cx)))
            prob = F.softmax(logit, dim=-1)
            action = prob.max(1, keepdim=True)[1][0]
            next_state, reward, done, _ = env.step(action.item())
            pre_hx, pre_cx = hx, cx
            state = next_state
            done = done or frame_length >= args.max_episode_length
            reward_ep.append(reward)
            # reward = max(min(reward, 1), -1)
            rewards += reward
            if done:
                print(f'epoch {episode} - frames {frame_length} - rewards {rewards}')
                state = env.reset()
                is_Terminal = True
                frame_length = 0