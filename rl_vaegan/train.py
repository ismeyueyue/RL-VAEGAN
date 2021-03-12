import sys
import os
sys.path.append(os.getcwd())
from rl_vaegan.utils import prepare_sub_folder, write_loss, get_config, write_2images
import argparse
from torch.autograd import Variable
from rl_vaegan.trainer import RL_VAEGAN
import torch.backends.cudnn as cudnn
import torch
import torch.nn.functional as F
import random as r
import numpy as np
import random
from collections import deque
import tensorboardX
import shutil
import ssl
import agent.my_optim as my_optim
from agent.envs import create_atari_env
from agent.model import ActorCritic
from attack.attacks import FGSM
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.cuda.set_device(0)

# if __name__ == "__main__":
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='rl_vaegan/config.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='rl_vaegan/', help="outputs path")
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max-episode-length', type=int, default=2000,help='maximum length of an episode (default: 1000000)')
parser.add_argument('--env-name', default='BreakoutDeterministic-v4', help='environment to train on (default: BreakoutDeterministic-v4 | PongDeterministic)')
parser.add_argument('--ft-setting', default='full-ft', help='fine-tuning setting: from-scratch|full-ft|random-output|partial-ft|partial-random-ft')
parser.add_argument('--attacker', default='fgsm', help='adversary attack algorithms: fgsm|rand_fgsm|cw2')
parser.add_argument('--epsilon-adv', type=float, default=0.003, help='epsilon perturbation for attack model.')

opts = parser.parse_args()

cudnn.benchmark = True

# Load experiment setting
config = get_config(opts.config)

torch.manual_seed(opts.seed)
env = create_atari_env(opts.env_name, opts)
trained_model = ActorCritic(env.observation_space.shape[0], env.action_space)

# load a pre-trained model according to the ft-setting
if opts.ft_setting == 'full-ft':
    if opts.env_name == 'BreakoutDeterministic-v4':
        fname = './agent/trained_model/breakout/11000.pth.tar'
    elif opts.env_name == 'PongDeterministic-v4':
        fname = './agent/trained_model/pong/4000.pth.tar'
    else:
        sys.exit('Only support Break or Pong')
        
    if os.path.isfile(fname):
        checkpoint = torch.load(fname)
        trained_model.load_state_dict(checkpoint['state_dict'])
        for param in trained_model.parameters():
            param.requires_grad = True
        print(f"{fname}\n Model was loaded successfully")

max_iter = config['max_iter']
config['vgg_model_path'] = opts.output_path

# Setup model and data loader
trainer = RL_VAEGAN(config)
trainer.cuda()

# Setup output folders
output_directory = opts.output_path + '/output/' + opts.env_name

checkpoint_directory, result_directory = prepare_sub_folder(output_directory)
print('checkpoint: ', checkpoint_directory)
print(f'attacker {opts.attacker} with epsilon {opts.epsilon_adv}')

# Start training
iterations =  0

trained_model.eval().cuda()
state = env.reset() #(3,80,80)
state = torch.from_numpy(state).unsqueeze(0).cuda()
episode_length = 1
epoch = 0
actions = deque(maxlen=100)
rewards = []
while True:
    epoch += 1
    episode_length = 1
    is_Terminal = False
    rewards = []
    while not is_Terminal:
        value, logit = trained_model(state) #(1,3,80,80)
        prob = F.softmax(logit, dim=-1)
        action = prob.multinomial(num_samples=1)[0]
        if opts.attacker ==  'fgsm':
            state_adv = FGSM(trained_model, name='a3c', eps=opts.epsilon_adv)._attack(state, action)

        images_a, images_b = state_adv.cuda().detach(), state.cuda().detach()
        dis_loss = trainer.dis_update(images_a, images_b, iterations, config)
        gen_loss = trainer.gen_update(images_a, images_b, iterations, config)
        torch.cuda.synchronize()
        trainer.update_learning_rate()

        state, reward, done, _ = env.step(action.item())
        done = done or episode_length >= opts.max_episode_length
        actions.append(action.item())

        # a quick hack to prevent the agent from stucking
        if actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            print(f'epoch {epoch} - steps {episode_length} - total rewards {np.sum(rewards) + reward}')
            actions.clear()
            state = env.reset()
        rewards.append(reward)
        state = torch.from_numpy(state).unsqueeze(0).cuda()
        if iterations % 50 == 0:
            print(f"Iteration: {iterations} | {max_iter}, \t dis_loss:{dis_loss} \t gen_loss:{gen_loss} ")
        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations)

        if (iterations + 1) % config['loss_save_iter'] == 0:
            trainer.save_loss(result_directory, iterations)

        episode_length += 1
        iterations += 1
        if iterations >= max_iter:
            sys.exit('Finish training')

        if done:
            is_Terminal = True
