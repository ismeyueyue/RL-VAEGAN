import numpy as np
from rl_vaegan.utils import get_config
from torch.autograd import Variable
from rl_vaegan.trainer import RL_VAEGAN
import torch.backends.cudnn as cudnn
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float()
    image_numpy = image_numpy.data.numpy()[0]
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


class TransferModel():
  def initialize(self, gan_dir, which_epoch, args):
    """
    initialize the model with the hyperparameters from the '.yaml' file
    :param gan_dir: the directory of the gan models
    :param which_epoch: the epoch the model was saved
    """
    print('TestModel initialize:', gan_dir, which_epoch)
    config = get_config('rl_vaegan/config.yaml')
    config['vgg_model_path'] = 'rl_vaegan'
    self.trainer = RL_VAEGAN(config)

    state_dict = torch.load("{}/gen_{}.pt".format(gan_dir, which_epoch), map_location=lambda storage, loc: storage)
    self.trainer.gen_a.load_state_dict(state_dict['a'])
    self.trainer.gen_b.load_state_dict(state_dict['b'])
    if torch.cuda.is_available():
      self.trainer.cuda()
    self.trainer.gen_a.eval()
    self.trainer.gen_b.eval()

  def transform(self, input):
    """
    translate input image to output domain
    :param input: input image
    """
    input = np.float32(input)  # (80,80,3)
    input = input.transpose((2, 0, 1))  # (3,80,80)
    final_data = ((torch.FloatTensor(input)/255.0)-0.5)*2

    final_data = Variable(final_data.view(1, final_data.size(
        0), final_data.size(1), final_data.size(2)))  # (1,3,80,80)
    if torch.cuda.is_available():
      final_data = final_data.cuda()

    content, _ = self.trainer.gen_a.encode(final_data)

    outputs = self.trainer.gen_b.decode(content)

    output_img = outputs[0].data.cpu().numpy()  # (1,3,80,80)
    out_img = np.uint8(
        255 * (np.transpose(output_img, (1, 2, 0)) / 2.0 + 0.5))  # (80,80,3)

    return out_img

  def transform_adv(self, input):
    """
    translate input image to output domain
    :param input: input image
    """
    # input = np.float32(input) # pixel (-1,1) (1,3,80,80)
    # input = input.transpose((2, 0, 1))
    # final_data = ((torch.FloatTensor(input))-0.5)*2 # pixel (-1,1)
    final_data = input

    final_data = Variable(final_data.view(final_data.size(
        0), final_data.size(1), final_data.size(2), final_data.size(3)))
    if torch.cuda.is_available():
      final_data = final_data.cuda()
    content, _ = self.trainer.gen_a.encode(final_data)  # (1,256,20,20)
    outputs = self.trainer.gen_b.decode(content)  # (1,3,80,80)
    # output_img = outputs[0].data.cpu().numpy()
    # out_img = (np.transpose(output_img, (1, 2, 0)) / 2.0 + 0.5) # pixel (0,1)
    # out_img = (output_img / 2.0 + 0.5) # pixel (0,1)

    return outputs
