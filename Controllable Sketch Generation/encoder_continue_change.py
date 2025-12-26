import torch
import torch.nn as nn
import torchvision
import time
import numpy as np
import random
from hyper_params import hp
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.autograd import Variable


class myencoder(nn.Module):
    def __init__(self, hps, ):
        super(myencoder, self).__init__()
        self.hps = hps

        self.k = 34
        self.z_size = hp.Nz

        self.global_ = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.de_mu = nn.Parameter(2 * torch.rand(self.k, self.z_size) - 1, requires_grad=False)
        self.de_sigma2 = nn.Parameter(torch.ones(size=[self.k, self.z_size], dtype=torch.float32), requires_grad=False)
        self.de_alpha = nn.Parameter(torch.ones(size=[self.k, 1], dtype=torch.float32)/self.k, requires_grad=False)
        self.fr = nn.Parameter(torch.ones(size=[self.k], dtype=torch.float32), requires_grad=False)
        
        # Initialize q_* parameters as copies of de_*
        self.q_mu = nn.Parameter(self.de_mu.detach().clone(), requires_grad=False)
        self.q_sigma2 = nn.Parameter(self.de_sigma2.detach().clone(), requires_grad=False)
        self.q_alpha = nn.Parameter(self.de_alpha.detach().clone(), requires_grad=False)
        # self.de_mu = self.de_mu.cuda()
        # self.de_alpha=self.de_alpha.cuda()
        # self.de_sigma2 = self.de_sigma2.cuda()
        self.db1_1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, dilation=1),
                                   nn.InstanceNorm2d(32),nn.ReLU(True))
        self.db1_2 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2, dilation=2),
                                   nn.InstanceNorm2d(32),nn.ReLU(True))
        self.db1_3 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=5, dilation=5),
                                   nn.InstanceNorm2d(32),nn.ReLU(True))

        self.db1_conv = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, dilation=1),
                                   nn.InstanceNorm2d(32),nn.ReLU(True))

        self.db2_1 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1),
                                   nn.InstanceNorm2d(64),nn.ReLU(True))
        self.db2_2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2, dilation=2),
                                   nn.InstanceNorm2d(64),nn.ReLU(True))
        self.db2_3 = nn.Sequential( nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=5, dilation=5),
                                   nn.InstanceNorm2d(64),nn.ReLU(True))

        self.db2_conv = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1),
                                      nn.InstanceNorm2d(64), nn.ReLU(True))

        self.downsampling1 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128),
                                           nn.ReLU(True))
        self.downsampling2 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),nn.InstanceNorm2d(256), nn.ReLU(True))

        self.upsampling4 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, 2, padding=1, output_padding=1),
                                         nn.InstanceNorm2d(128),
                                         nn.ReLU(True))
        self.upsampling3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
                                         nn.InstanceNorm2d(64)
                                         , nn.ReLU(True))

        self.upsampling2 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1),
                                         nn.InstanceNorm2d(32)
                                         , nn.ReLU(True))

        self.upsampling1 = nn.Sequential(nn.ConvTranspose2d(32, 1, 3, 2, padding=1, output_padding=1))

        self.fc_mu = nn.Linear(256*9, self.hps.Nz)
        self.fc_sigma = nn.Linear(256*9, self.hps.Nz)
        self.fc = nn.Sequential(nn.Linear(self.hps.Nz, 256*9), nn.ReLU(True))
        self.mseloss =nn.MSELoss()

    def forward(self, graph):
        # print("demu",self.de_mu.mean())
        image = graph[:,0].view(graph.shape[0],1,graph.shape[-1],graph.shape[-1])
        gt = graph[:,1].view(graph.shape[0],1,graph.shape[-1],graph.shape[-1])

        shape = image.shape
        self.batch_size = shape[0]
        
        # Ensure q_* parameters are on the same device as input
        if self.q_mu.device != graph.device:
            self.q_mu.data = self.q_mu.data.to(graph.device)
            self.q_sigma2.data = self.q_sigma2.data.to(graph.device)
            self.q_alpha.data = self.q_alpha.data.to(graph.device)
            self.de_mu.data = self.de_mu.data.to(graph.device)
            self.de_sigma2.data = self.de_sigma2.data.to(graph.device)
            self.de_alpha.data = self.de_alpha.data.to(graph.device)
            self.fr.data = self.fr.data.to(graph.device)

        top = self.get_feature(image)
        # latent_code = self.pooling(top).view(shape[0],512*4)

        latent_code = top.view(shape[0], 256*9)

        mu = self.fc_mu(latent_code)
        sigma = self.fc_sigma(latent_code)

        #sigma_e = torch.exp(sigma / 2.)
        sigma2 = F.softplus(sigma) + 1e-10
        sigma_e = torch.sqrt(sigma2)
        z_size = mu.size()
        if mu.get_device() != -1:  # not in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size)).cuda(mu.get_device())
        else:  # in cpu
            n = torch.normal(torch.zeros(z_size), torch.ones(z_size))
        # sample z
        z = mu + sigma_e * n
        down = self.rec_image(z)

        self.p_mu = mu
        self.p_sigma2 = sigma_e**2
        self.p_alpha, self.gau_label = self.calculate_posterior(z, self.q_mu, self.q_sigma2, self.q_alpha)

        #print(self.p_alpha,self.gau_label)
        if self.global_ > hp.br+500:
            if (self.global_ % 3) == 0:
                q_mu, q_sigma2, q_alpha = self.em(self.p_mu, self.p_sigma2, self.p_alpha, self.q_mu,
                                                                 self.q_sigma2, self.q_alpha)
            else:
                q_mu, q_sigma2, q_alpha = self.rpcl(self.p_mu, self.p_sigma2, self.p_alpha, self.q_mu,
                                                                 self.q_sigma2, self.q_alpha)
        else:

            q_mu, q_sigma2, q_alpha = self.em(self.p_mu, self.p_sigma2, self.p_alpha,self.q_mu,
                                                                 self.q_sigma2, self.q_alpha)

        self.alpha_loss = self.get_alpha_loss(self.p_alpha, Variable(q_alpha.data,requires_grad=False))
        self.gaussian_loss = self.get_gaussian_loss(self.p_alpha, self.p_mu, self.p_sigma2, Variable(q_mu.data,requires_grad=False),
                Variable(q_sigma2.data,requires_grad=False))
        rpcl_loss = self.gaussian_loss+self.alpha_loss
        
        # Update q_* parameters with the new values for progressive adaptation
        self.q_mu.data = q_mu.data
        self.q_sigma2.data = q_sigma2.data
        self.q_alpha.data = q_alpha.data
        
        return z, mu, sigma, self.mseloss(down,image)*48*48, (rpcl_loss,self.alpha_loss,self.gaussian_loss), (q_mu, q_sigma2, q_alpha)

    def get_feature(self, image):
        x = self.db1_1(image)+self.db1_2(image)+self.db1_3(image)
        x = self.db1_conv(x)
        x = self.db2_1(x) + self.db2_2(x) + self.db2_3(x)
        x = self.db2_conv(x)
        x = self.downsampling1(x)
        top = self.downsampling2(x)
        #top = torch.tanh(top)
        return top

    def rec_image(self,top):
        feature = self.fc(top).view(-1,256,3,3)
        y3 = self.upsampling4(feature)
        y2 = self.upsampling3(y3)
        y1 = self.upsampling2(y2)
        y0 = self.upsampling1(y1)
        y0 = torch.tanh(y0)
        return y0


    def assign(self, data):
        self.de_mu.data = data[0]
        self.de_alpha.data = data[2]
        self.de_sigma2.data = data[1]
        # Also update q_* parameters to maintain consistency
        self.q_mu.data = data[0]
        self.q_alpha.data = data[2]
        self.q_sigma2.data = data[1]

    def calculate_prob(self, x, q_mu, q_sigma2):
        """ Calculate the probabilistic density """
        # x [batch_size, Nz] q[k,Nz]
        mu = q_mu.view(1, self.k, self.z_size).repeat(self.batch_size, 1, 1)
        sigma2 = q_sigma2.view(1, self.k, self.z_size).repeat(self.batch_size, 1, 1)
        x = x.view(self.batch_size, 1, self.z_size).repeat(1, self.k, 1)

        log_exp_part = -0.5 * torch.sum(torch.div(torch.square(x - mu), 1e-30 + sigma2), dim=2)
        log_frac_part = torch.sum(torch.log(torch.sqrt(sigma2 + 1e-30)), dim=2)
        log_prob = log_exp_part - log_frac_part - float(self.z_size) / 2. * torch.tensor(
            np.log([2. * 3.1416]), device=x.device, dtype=x.dtype)
        return torch.exp(log_prob.to(torch.float64))

    def calculate_posterior(self, y, q_mu, q_sigma2, q_alpha):
        prob = self.calculate_prob(y, q_mu, q_sigma2)
        #print(prob[0])
        temp = torch.multiply(torch.transpose(q_alpha, 1, 0).repeat(self.batch_size, 1), prob)
        sum_temp = torch.sum(temp, dim=1, keepdim=True).repeat(1, self.k)
        #print(prob[0:5])
        gamma = torch.clamp((torch.div(temp, 1e-300 + sum_temp)), 1e-5, 1.)
        # 对gamma标准化,第k个部件和目标之间的距离和
        gamma_st = gamma / (1e-10 + (torch.sum(gamma, dim=1, keepdim=True)).repeat(1, self.k))
        return gamma_st, gamma_st.argmax(dim=1)

    def get_alpha_loss(self, p_alpha, q_alpha):
        # 计算KL散度
        p_alpha = p_alpha.view(self.batch_size, self.k)
        q_alpha = q_alpha.view(1, self.k).repeat(self.batch_size, 1)
        return torch.sum(torch.mean(p_alpha * torch.log(torch.div(p_alpha, q_alpha + 1e-10) + 1e-10), dim=0))

    def get_gaussian_loss(self, p_alpha, p_mu, p_sigma2, q_mu, q_sigma2):
        p_alpha = p_alpha.view(self.batch_size, self.k)
        p_mu = p_mu.view(self.batch_size, 1, self.z_size).repeat(1, self.k, 1)
        p_sigma2 = p_sigma2.view(self.batch_size, 1, self.z_size).repeat(1, self.k, 1)
        q_mu = q_mu.view(1, self.k, self.z_size).repeat(self.batch_size, 1, 1)
        q_sigma2 = q_sigma2.view(1, self.k, self.z_size).repeat(self.batch_size, 1, 1)

        return torch.mean(torch.sum(0.5 * p_alpha * torch.sum(
            torch.log(q_sigma2 + 1e-10) + (p_sigma2 + (p_mu - q_mu) ** 2) / (q_sigma2 + 1e-10)
            - 1.0 - torch.log(p_sigma2 + 1e-10), dim=2), dim=1))

    def em(self, y, en_sigma2, gamma, q_mu_old, q_sigma2_old, q_alpha_old):
        en_sigma2 = en_sigma2.view(self.batch_size, 1, self.z_size).repeat(1, self.k, 1)

        sum_gamma = torch.sum(gamma, dim=0).unsqueeze(1).repeat(1, self.z_size)

        temp_y = y.unsqueeze(1).repeat(1, self.k, 1)
        q_mu_new = torch.sum(temp_y * gamma.unsqueeze(2).repeat(1, 1, self.z_size), dim=0) / (sum_gamma + 1e-10)

        q_sigma2_new = torch.sum(
            (torch.square(temp_y - q_mu_new.unsqueeze(0).repeat(self.batch_size, 1, 1)) + en_sigma2)
            * gamma.unsqueeze(2).repeat(1, 1, self.z_size), dim=0) \
                       / (sum_gamma + 1e-10)

        q_alpha_new =torch.mean(gamma, dim=0).unsqueeze(1)
        factor = 0.95
        #if self.global_%1000 ==0 :
           # self.de_mu.data = 2 * torch.rand(self.k, self.z_size).cuda() - 1
            #self.de_sigma2.data = torch.ones(size=[self.k, self.z_size]).cuda()
            #self.de_alpha.data = torch.ones(size=[self.k, 1]).cuda()
            #factor = 1

        #elif self.global_%100 == 0:
            #factor=0.0
        q_mu = q_mu_old * factor + q_mu_new * (1 - factor)
        q_sigma2 = torch.clamp(q_sigma2_old * factor + q_sigma2_new * (1 - factor), 1e-10, 1e10)
        q_alpha = torch.clamp(q_alpha_old * factor + q_alpha_new * (1 - factor), 0., 1.)
        q_alpha_st = q_alpha / torch.sum(q_alpha)

        return q_mu, q_sigma2, q_alpha_st

    def rpcl(self, y, en_sigma2, gamma, q_mu_old, q_sigma2_old, q_alpha_old):
        en_sigma2 = en_sigma2.view(self.batch_size, 1, self.z_size).repeat(1, self.k, 1)
        penalize = 1e-4 # De-learning rate
        # penalize_2 =1e-6
        # y:[batch_size,Nz]， argmax返回的时索引值，而不是具体的某个值
        # temp_y [batch_size,k,Nz]
        fr = self.fr.data/self.fr.data.sum()
        gamma = gamma/fr.unsqueeze(0).repeat(gamma.shape[0],1)
        temp_y = y.unsqueeze(1).repeat(1, self.k, 1)

        winner = F.one_hot(torch.argmax(gamma, dim=1), self.k)
        rival = F.one_hot(torch.argmax(gamma - gamma * winner, dim=1), self.k)
        self.fr.data = self.fr.data+winner.sum(dim=0)

        gamma_rpcl = winner - penalize * rival  # -penalize_2*rival_2  # [batch_size,  k]
        sum_gamma_rpcl = torch.sum(gamma_rpcl, dim=0).unsqueeze(1).repeat(1, self.z_size)

        q_mu_new = torch.sum(temp_y * gamma_rpcl.unsqueeze(2).repeat(1, 1, self.z_size), dim=0) / (
                sum_gamma_rpcl + 1e-10)

        q_sigma2_new = torch.sum(
            (torch.square(temp_y - q_mu_new.unsqueeze(0).repeat(self.batch_size, 1, 1)) + en_sigma2)
            * gamma_rpcl.unsqueeze(2).repeat(1, 1, self.z_size), dim=0) \
                       / (sum_gamma_rpcl + 1e-10)

        q_alpha_new = torch.mean(gamma_rpcl, dim=0).unsqueeze(1)
        factor = 0.95

        q_mu = q_mu_old * factor + q_mu_new * (1 - factor)
        q_sigma2 = torch.clamp(q_sigma2_old * factor + q_sigma2_new * (1 - factor), 1e-10, 1e10)
        q_alpha = torch.clamp(q_alpha_old * factor + q_alpha_new * (1 - factor), 0., 1.)
        q_alpha_st = q_alpha / torch.sum(q_alpha)

        return q_mu, q_sigma2, q_alpha_st

