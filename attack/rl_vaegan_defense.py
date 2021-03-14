import time
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from agent.envs import create_atari_env
from agent.model import ActorCritic, ActorCritic_Substitude
from os import listdir
from attack.attacks import FGSM, RandFGSM, CW2
import os
import random
import sys
import pickle
from os.path import isfile, join

def transfer_defense(rank, args, shared_model, counter):
    torch.manual_seed(args.seed + rank)

    rl_vaegan_path = 'rl_vaegan/output/' + args.env_name + '/checkpoints'

    env = create_atari_env(args.env_name, args)

    '''load trained RL-VAEGAN model'''
    import rl_vaegan.transfer as t
    translate_model = t.TransferModel()
    translate_model.initialize(rl_vaegan_path, arg.which_epoch, args)
    
    env.seed(args.seed + rank)
    if args.black_box_attack:
        print('Black Box Attack')
        model = ActorCritic_Substitude(env.observation_space.shape[0], env.action_space)
    else:
        print('White Box Attack')
        model = ActorCritic(env.observation_space.shape[0], env.action_space)
    
    if args.test_attacker == 'rand_fgsm':
        test_alpha_adv = args.test_epsilon_adv * 0.5

    print(f'FGSM test on attacker: {args.test_attacker} - epsilon: {args.test_epsilon_adv}')

    if args.test_attacker == 'cw2':
        test_iteration = 30
    else:
        test_iteration = 30

    state = env.reset()
    state = torch.from_numpy(state).unsqueeze(0).cuda()
    reward_sum = 0
    done = True
    episode_length = 0
    total_step = 0   
    actions = deque(maxlen=100)
    reward_ep = []
    for epoch in range(test_iteration):
        model.load_state_dict(shared_model.state_dict())
        model.eval().cuda()
        rewards = []
        # for step in range(args.num_steps):
        is_Terminal = False
        while not is_Terminal:
            episode_length += 1
            total_step += 1
            with torch.no_grad():
                value, logit = model(state)
            prob = F.softmax(logit, dim=-1)
            action = prob.multinomial(num_samples=1)[0]
            '''adversarial attack'''
            if args.variation == 'adversary':
                if args.test_attacker ==  'fgsm':
                    # args.epsilon_adv = random.randint(1,5) * 0.001
                    state_adv = FGSM(model, name='a3c', eps=args.test_epsilon_adv)._attack(state, action) #(1,3,80,80)
                elif args.test_attacker == 'rand_fgsm':
                    # args.epsilon_adv = random.randint(2,5) * 0.001
                    # args.alpha_adv = args.epsilon_adv * 0.5
                    state_adv = RandFGSM(model, name='a3c', eps=args.test_epsilon_adv, alpha=test_alpha_adv)._attack(state, action)
                elif args.test_attacker == 'cw2':
                    state_adv = CW2(model, name='a3c')._attack(state, action, env.action_space.n)
                else:
                    sys.exit('with attacker in (FGSM | Rand+FGSM | CW2) !')

            '''rl_vaegan style transfer defense'''
            state_def = translate_model.transform_adv(state_adv)

            with torch.no_grad():
                value_def, logit_def = model(state_def)

            prob_def = F.softmax(logit_def, dim=-1)
            action_def = prob_def.multinomial(num_samples=1)[0]

            state, reward, done, _ = env.step(action_def.item())
            done = done or episode_length >= args.max_episode_length
            actions.append(action_def.item())
            # a quick hack to prevent the agent from stucking
            if actions.count(actions[0]) == actions.maxlen:
                done = True
            if done:
                # print(episode_length)
                print(f'epoch {epoch} | {test_iteration} - steps {episode_length} - total rewards {np.sum(rewards) + reward}')
                reward_ep.append(np.sum(rewards) + reward)
                print('episode rewards:', reward_ep, 'avg: ', np.sum(reward_ep) / len(reward_ep))
                episode_length = 0
                actions.clear()
                state = env.reset()
            rewards.append(reward)
            state = torch.from_numpy(state).unsqueeze(0).cuda()
            if done:
                is_Terminal = True

    print('episode rewards:', reward_ep, 'avg: ', np.sum(reward_ep) / len(reward_ep))