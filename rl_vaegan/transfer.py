import torch
from torch.autograd import Variable

from rl_vaegan.trainer import RL_VAEGAN
from rl_vaegan.utils import get_config

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class TransferModel():
    def initialize(self, vaegan_dir, which_epoch, args):
        print('TestModel initialize:', vaegan_dir, which_epoch)
        config = get_config('rl_vaegan/config.yaml')
        config['vgg_model_path'] = 'rl_vaegan'
        self.trainer = RL_VAEGAN(config)

        state_dict = torch.load("{}/gen_{}.pt".format(vaegan_dir, which_epoch),
                                map_location=lambda storage, loc: storage)
        self.trainer.gen_a.load_state_dict(state_dict['a'])
        self.trainer.gen_b.load_state_dict(state_dict['b'])
        if torch.cuda.is_available():
            self.trainer.cuda()
        self.trainer.gen_a.eval()
        self.trainer.gen_b.eval()

    def transform_adv(self, input):
        final_data = input

        final_data = Variable(
            final_data.view(final_data.size(0), final_data.size(1),
                            final_data.size(2), final_data.size(3)))
        if torch.cuda.is_available():
            final_data = final_data.cuda()
        content, _ = self.trainer.gen_a.encode(final_data)  # (1,256,20,20)
        outputs = self.trainer.gen_b.decode(content)  # (1,3,80,80)

        return outputs
