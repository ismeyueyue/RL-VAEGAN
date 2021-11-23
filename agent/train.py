import os
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from agent.envs import create_atari_env
from agent.model import ActorCritic


def prepare_sub_folder(output_directory):
    result_directory = os.path.join(output_directory, 'results')
    if not os.path.exists(result_directory):
        print("Creating directory: {}".format(result_directory))
        os.makedirs(result_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, result_directory


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


def train(rank, args, shared_model, counter, lock, optimizer=None):
    print('Train with A3C')
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name, args)
    env.seed(args.seed + rank)

    model = ActorCritic(env.observation_space.shape[0], env.action_space)

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    model.train()
    output_directory = 'outputs/' + args.env_name
    checkpoint_directory, result_directory = prepare_sub_folder(
        output_directory)
    print(f'checkpoint directory {checkpoint_directory}')
    time.sleep(10)
    state = env.reset()
    state = torch.from_numpy(state)
    done = True
    episode_length = 0
    total_step = 0
    rewards_ep = []
    policy_loss_ep = []
    value_loss_ep = []
    for epoch in range(100000000):
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())

        values = []
        log_probs = []
        rewards = []
        entropies = []

        # for step in range(args.num_steps):
        is_Terminal = False
        while not is_Terminal:
            episode_length += 1
            total_step += 1
            value, logit = model(state.unsqueeze(0))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()
            log_prob = log_prob.gather(1, action)

            state, reward, done, _ = env.step(action.numpy())

            done = done or episode_length >= args.max_episode_length
            reward = max(min(reward, 1), -1)

            with lock:
                counter.value += 1

            if done:
                # print(episode_length)
                print(
                    f'epoch {epoch} - steps {total_step} - total rewards {np.sum(rewards) + reward}'
                )
                total_step = 1
                episode_length = 0
                state = env.reset()

            state = torch.from_numpy(state)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                rewards_ep.append(np.sum(rewards))
                is_Terminal = True
                # break

        R = torch.zeros(1, 1)
        if not done:
            value, _ = model(state.unsqueeze(0))
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1] - values[i]
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()

        policy_loss_ep.append(policy_loss.detach().numpy()[0, 0])
        value_loss_ep.append(value_loss.detach().numpy()[0, 0])

        (policy_loss + args.value_loss_coef * value_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        ensure_shared_grads(model, shared_model)
        optimizer.step()

        if epoch % 1000 == 0:
            torch.save({'state_dict': model.state_dict()},
                       checkpoint_directory + '/' + str(epoch) + ".pth.tar")
            with open(result_directory + '/' + str(epoch) + '_rewards.pkl',
                      'wb') as f:
                pickle.dump(rewards_ep, f)
            with open(result_directory + '/' + str(epoch) + '_policy_loss.pkl',
                      'wb') as f:
                pickle.dump(policy_loss_ep, f)
            with open(result_directory + '/' + str(epoch) + '_value_loss.pkl',
                      'wb') as f:
                pickle.dump(value_loss_ep, f)

        if episode_length >= 10000000:
            break

    torch.save({
        'state_dict': model.state_dict(),
    }, checkpoint_directory + '/Last' + ".pth.tar")
