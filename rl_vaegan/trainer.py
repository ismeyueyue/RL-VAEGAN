from rl_vaegan.networks import MsImageDis, VAEGen
from rl_vaegan.utils import weights_init, get_model_list, vgg_preprocess, load_vgg16, get_scheduler
from torch.autograd import Variable
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle

class RL_VAEGAN(nn.Module):
    def __init__(self, hyperparameters):
        super(RL_VAEGAN, self).__init__()
        self.loss_dis_ep = []
        self.loss_gen_ep = []

        self.batch_size = hyperparameters['batch_size']

        lr = hyperparameters['lr']
        # Initiate the networks
        self.gen_a = VAEGen(hyperparameters['input_dim_a'], hyperparameters['gen'])  # vae for style a
        self.gen_b = VAEGen(hyperparameters['input_dim_b'], hyperparameters['gen'])  # vae for style b
        self.dis_a = MsImageDis(hyperparameters['input_dim_a'], hyperparameters['dis'])  # discriminator for style a
        self.dis_b = MsImageDis(hyperparameters['input_dim_b'], hyperparameters['dis'])  # discriminator for style b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)

        self.enc_weight_sharing = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.dec_weight_sharing = nn.Conv2d(3, 3, kernel_size=(1, 1), stride=(1, 1), bias=False)

        # Setup the optimizers
        beta1 = hyperparameters['beta1']
        beta2 = hyperparameters['beta2']
        dis_params = list(self.dis_a.parameters()) + list(self.dis_b.parameters())

        gen_params = list(self.gen_a.parameters()) + list(self.gen_b.parameters()) + list(self.enc_weight_sharing.parameters())

        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                        lr=lr, betas=(beta1, beta2), weight_decay=hyperparameters['weight_decay'])
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters)

        # Network weight initialization
        self.apply(weights_init(hyperparameters['init']))
        self.dis_a.apply(weights_init('gaussian'))
        self.dis_b.apply(weights_init('gaussian'))

        # Load VGG model if needed
        if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
            self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
            self.vgg.eval()
            for param in self.vgg.parameters():
                param.requires_grad = False

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self, x_a, x_b):
        self.eval()
        h_a, _ = self.gen_a.encode(x_a)
        h_b, _ = self.gen_b.encode(x_b)
        x_ba = self.gen_a.decode(h_b)
        x_ab = self.gen_b.decode(h_a)
        self.train()
        return x_ab, x_ba

    def __compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

    # p_logit: [batch, class_num]
    # q_logit: [batch, class_num]
    def __kl_divergence(self, mu_1, mu_2):
        mu = torch.pow((mu_1 - mu_2), 2)
        bi_direction_loss = torch.mean(mu)
        return bi_direction_loss

    def compute_constactive_loss(self, adv, model):
        jacobi_x = model.enc.compute_jacobi_x(adv)[0]
        constactive_loss = torch.sum(jacobi_x ** 2)
        return  constactive_loss

    def gen_update(self, x_a, x_b, iterations, hyperparameters):
        self.gen_opt.zero_grad()
        # encode 
        # reparameterizetion sigma = 1 // multivariate Gaussian distribution with mean = hiddens and std_dev = all ones
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)

        # Weight sharing in encode_last layer
        h_a = self.enc_weight_sharing(h_a)
        h_b = self.enc_weight_sharing(h_b)

        # decode (within style)
        x_a_recon = self.gen_a.decode(h_a + n_a)
        x_b_recon = self.gen_b.decode(h_b + n_b)

        # Weight sharing in decode_last_layer
        x_a_recon = self.dec_weight_sharing(x_a_recon)
        x_b_recon = self.dec_weight_sharing(x_b_recon)

        # decode (cross style)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)

        # encode again
        h_b_recon, n_b_recon = self.gen_a.encode(x_ba)
        h_a_recon, n_a_recon = self.gen_b.encode(x_ab)

        # decode again (if needed)
        x_aba = self.gen_a.decode(h_a_recon + n_a_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None
        x_bab = self.gen_b.decode(h_b_recon + n_b_recon) if hyperparameters['recon_x_cyc_w'] > 0 else None

        '''constractive loss'''
        self.loss_gen_ca_x_a = self.compute_constactive_loss(x_a, self.gen_a)

        '''bi reconstruction loss'''
        h_a_x_a_recon, _  = self.gen_a.encode(x_a_recon)
        h_b_x_b_recon, _  = self.gen_a.encode(x_b_recon)
        self.loss_bi_direction_x_a = self.__kl_divergence(h_a_x_a_recon, h_a)
        self.loss_bi_direction_x_b = self.__kl_divergence(h_b_x_b_recon, h_b)

        '''reconstruction loss'''
        self.loss_gen_recon_x_a = self.recon_criterion(x_a_recon, x_a)
        self.loss_gen_recon_x_b = self.recon_criterion(x_b_recon, x_b)
        self.loss_gen_recon_kl_a = self.__compute_kl(h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(h_b)

        '''recon cycle consistency constraint'''
        self.loss_gen_cyc_x_a = self.recon_criterion(x_aba, x_a)
        self.loss_gen_cyc_x_b = self.recon_criterion(x_bab, x_b)

        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(h_b_recon)
        
        '''GAN loss'''
        self.loss_gen_adv_a = self.dis_a.calc_gen_loss(x_ba)
        self.loss_gen_adv_b = self.dis_b.calc_gen_loss(x_ab)

        '''style-invariant perceptual loss'''
        self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, x_ba, x_b) if hyperparameters['vgg_w'] > 0 else 0
        self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, x_ab, x_a) if hyperparameters['vgg_w'] > 0 else 0

        # total loss
        self.loss_gen_total = hyperparameters['gan_w'] * self.loss_gen_adv_a + \
                            hyperparameters['gan_w'] * self.loss_gen_adv_b + \
                            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_a + \
                            hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                            hyperparameters['recon_x_w'] * self.loss_gen_recon_x_b + \
                            hyperparameters['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                            hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                            hyperparameters['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                            hyperparameters['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab + \
                            hyperparameters['vgg_w'] * self.loss_gen_vgg_a + \
                            hyperparameters['recon_x_bi_direction_w'] * self.loss_bi_direction_x_a + \
                            hyperparameters['recon_x_bi_direction_w'] * self.loss_bi_direction_x_b + \
                            hyperparameters['vgg_w'] * self.loss_gen_vgg_b + hyperparameters['constactive_w'] * self.loss_gen_ca_x_a


        self.loss_gen_total.backward()
        self.gen_opt.step()
        if iterations % 20 == 0:
            self.loss_gen_ep.append(self.loss_gen_total.detach().cpu().item())
        return self.loss_gen_total.detach().cpu().item()

    def compute_vgg_loss(self, vgg, img, target):
        img_vgg = vgg_preprocess(img)
        target_vgg = vgg_preprocess(target)
        img_fea = vgg(img_vgg)
        target_fea = vgg(target_vgg)
        # print(f'img-{img.shape} | target-{target.shape} | img_fea-{img_fea.shape} | target_fea-{target_fea.shape}')
        return torch.mean((self.instancenorm(img_fea) - self.instancenorm(target_fea)) ** 2)

    def dis_update(self, x_a, x_b, iterations, hyperparameters):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.gen_a.encode(x_a)
        h_b, n_b = self.gen_b.encode(x_b)
        # decode (cross style)
        x_ba = self.gen_a.decode(h_b + n_b)
        x_ab = self.gen_b.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.dis_a.calc_dis_loss(x_ba.detach(), x_a)
        self.loss_dis_b = self.dis_b.calc_dis_loss(x_ab.detach(), x_b)
        self.loss_dis_total = hyperparameters['gan_w'] * self.loss_dis_a + hyperparameters['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()
        self.dis_opt.step()
        if iterations % 20 == 0:
            self.loss_dis_ep.append(self.loss_dis_total.detach().cpu().item())
        return self.loss_dis_total.detach().cpu().item()

    def update_learning_rate(self):
        # lr_scheduler.step()
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()

    def resume(self, checkpoint_dir, hyperparameters):
        # Load generators
        last_model_name = get_model_list(checkpoint_dir, "gen")
        print (last_model_name)
        state_dict = torch.load(last_model_name)
        print (state_dict.keys())
        self.gen_a.load_state_dict(state_dict['a'])
        self.gen_b.load_state_dict(state_dict['b'])
        iterations = int(last_model_name[-11:-3])
        # Load discriminators
        last_model_name = get_model_list(checkpoint_dir, "dis")
        state_dict = torch.load(last_model_name)
        self.dis_a.load_state_dict(state_dict['a'])
        self.dis_b.load_state_dict(state_dict['b'])
        # Load optimizers
        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.dis_opt.load_state_dict(state_dict['dis'])
        self.gen_opt.load_state_dict(state_dict['gen'])
        # Reinitilize schedulers
        self.dis_scheduler = get_scheduler(self.dis_opt, hyperparameters, iterations)
        self.gen_scheduler = get_scheduler(self.gen_opt, hyperparameters, iterations)
        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations):
        # Save generators, discriminators, and optimizers
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        dis_name = os.path.join(snapshot_dir, 'dis_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        torch.save({'a': self.gen_a.state_dict(), 'b': self.gen_b.state_dict()}, gen_name)
        torch.save({'a': self.dis_a.state_dict(), 'b': self.dis_b.state_dict()}, dis_name)
        torch.save({'gen': self.gen_opt.state_dict(), 'dis': self.dis_opt.state_dict()}, opt_name)
    
    def save_loss(self, snapshot_dir, iterations):
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir)
        with open(snapshot_dir + '/' + str(iterations + 1) + '_dis_loss.pkl', 'wb') as f:
            pickle.dump(self.loss_dis_ep, f)
        with open(snapshot_dir + '/' + str(iterations + 1) + '_gen_loss.pkl', 'wb') as f:
            pickle.dump(self.loss_gen_ep, f)       
