import cv2
import gym
import numpy as np
from gym.spaces.box import Box
import os

change_counter = 1
positions = [20, 40, 60]
current_position = 0

img_counter = 1
args = None


def get_img_counter():
    return img_counter


# Taken from https://github.com/openai/universe-starter-agent
def create_atari_env(env_id, arguments, test_gan=False, gan_file=None):
    global args
    args = arguments

    if test_gan:
        import rl_vaegan.transfer_defense as t
        model = t.TransferModel()
        gan_path = args.gan_models_path + '/' + args.gan_dir
        model.initialize(gan_path, gan_file)

        env = gym.make(env_id)
        env = AtariRescale42x42(env, test_gan, gan_file)
        return env, model
    else:
        env = gym.make(env_id)
        env = AtariRescale42x42(env, test_gan, gan_file)
        return env


def _process_frame42(frame, test_gan, gan_file):
    global img_counter
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame_adv = frame.copy()

    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    # trans this data to [-1, 1]
    frame = (frame * 2.0) - 1.0
    frame = np.moveaxis(frame, -1, 0)
    return frame


class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None, test_gan=None, gan_file=None):
        super(AtariRescale42x42, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [3, 80, 80], dtype=np.float32)
        self.test_gan = test_gan
        self.gan_file = gan_file

    def _observation(self, observation):
        return _process_frame42(observation, self.test_gan, self.gan_file)
