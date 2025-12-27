import os

from hyper_params import hp
import numpy as np
import matplotlib.pyplot as plt
import PIL
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch import optim
from encoder import myencoder
from decoder import DecoderRNN
from utils.sketch_processing import make_graph


################################# load and prepare data
class SketchesDataset:
    def __init__(self, path: str, category: list, mode="train"):
        self.sketches = None
        self.sketches_normed = None
        self.max_sketches_len = 0
        self.path = path
        self.category = category
        self.mode = mode

        tmp_sketches = []
        for c in self.category:
            dataset = np.load(os.path.join(self.path, c), encoding='latin1', allow_pickle=True)
            tmp_sketches.append(dataset[self.mode])
            print(f"dataset: {c} added.")
        data_sketches = np.concatenate(tmp_sketches)
        print(f"length of trainset: {len(data_sketches)}")

        data_sketches = self.purify(data_sketches)  # data clean.  # remove toolong and too stort sketches.
        self.sketches = data_sketches.copy()
        self.sketches_normed = self.normalize(data_sketches)
        self.Nmax = self.max_size(data_sketches)  # max size of a sketch.

    def max_size(self, sketches):
        """返回所有sketch中 转折最多的一个sketch"""
        sizes = [len(sketch) for sketch in sketches]
        return max(sizes)

    def purify(self, sketches):
        data = []
        for sketch in sketches:
            if hp.max_seq_length >= sketch.shape[0] > hp.min_seq_length:  # remove small and too long sketches.
                sketch = np.minimum(sketch, 1000)  # remove large gaps.
                sketch = np.maximum(sketch, -1000)
                sketch = np.array(sketch, dtype=np.float32)  # change it into float32
                data.append(sketch)
        return data

    def calculate_normalizing_scale_factor(self, sketches):
        data = []
        for sketch in sketches:
            for stroke in sketch:
                data.append(stroke)
        return np.std(np.array(data))

    def normalize(self, sketches):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        data = []
        scale_factor = self.calculate_normalizing_scale_factor(sketches)
        for sketch in sketches:
            sketch[:, 0:2] /= scale_factor
            data.append(sketch)
        return data

    def make_batch(self, batch_size):
        """
        :param batch_size:
        :return:
        """
        batch_idx = np.random.choice(len(self.sketches_normed), batch_size)
        batch_sketches = [self.sketches_normed[idx] for idx in batch_idx]
        batch_sketches_graphs = [self.sketches[idx] for idx in batch_idx]
        sketches = []
        lengths = []
        graphs = []  # (batch_size * graphs_num_constant, x, y)
        adjs = []
        index = 0
        for _sketch in batch_sketches:
            len_seq = len(_sketch[:, 0])  # sketch
            new_sketch = np.zeros((self.Nmax, 5))  # new a _sketch, all length of sketch in size is Nmax.
            new_sketch[:len_seq, :2] = _sketch[:, :2]

            # set p into one-hot.
            new_sketch[:len_seq - 1, 2] = 1 - _sketch[:-1, 2]
            new_sketch[:len_seq, 3] = _sketch[:, 2]

            # len to Nmax set as 0,0,0,0,1
            new_sketch[(len_seq - 1):, 4] = 1
            new_sketch[len_seq - 1, 2:4] = 0  # x, y, 0, 0, 1
            lengths.append(len(_sketch[:, 0]))  # lengths is _sketch length, not new_sketch length.
            sketches.append(new_sketch)
            index += 1

        for _each_sketch in batch_sketches_graphs:
            _graph_tensor, _adj_matrix = make_graph(_each_sketch, graph_num=hp.graph_number,
                                                    graph_picture_size=hp.graph_picture_size, mask_prob=hp.mask_prob)
            graphs.append(_graph_tensor)
            adjs.append(_adj_matrix)

        if hp.use_cuda:
            batch = torch.from_numpy(np.stack(sketches, 1)).cuda().float()  # (Nmax, batch_size, 5)
            graphs = torch.from_numpy(np.stack(graphs, 0)).cuda().float()  # (batch_size, len, 5)
            adjs = torch.from_numpy(np.stack(adjs, 0)).cuda().float()

        else:
            batch = torch.from_numpy(np.stack(sketches, 1)).float()  # (Nmax, batch_size, 5)
            graphs = torch.from_numpy(np.stack(graphs, 0)).float()
            adjs = torch.from_numpy(np.stack(adjs, 0)).float()

        return batch, lengths, graphs, adjs


sketch_dataset = SketchesDataset(hp.data_location, hp.category, "train")
hp.Nmax = sketch_dataset.Nmax


def sample_bivariate_normal(mu_x: torch.Tensor, mu_y: torch.Tensor,
                            sigma_x: torch.Tensor, sigma_y: torch.Tensor,
                            rho_xy: torch.Tensor, greedy=False):
    mu_x = mu_x.item()
    mu_y = mu_y.item()
    sigma_x = sigma_x.item()
    sigma_y = sigma_y.item()
    rho_xy = rho_xy.item()
    # inputs must be floats
    if greedy:
        return mu_x, mu_y
    mean = [mu_x, mu_y]

    sigma_x *= np.sqrt(hp.temperature)
    sigma_y *= np.sqrt(hp.temperature)

    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
           [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def make_image(sequence, epoch, name='_output_'):
    strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0] + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                    canvas.tostring_rgb())
    name = f"{hp.model_save}" + str(epoch) + name + '.jpg'
    pil_image.save(name, "JPEG")
    plt.close("all")


################################# encoder and decoder modules


class Model:
    def __init__(self):
        if hp.use_cuda:
            # self.encoder: nn.Module = EncoderGCN(hp.graph_number, hp.graph_picture_size, hp.out_f_num, hp.Nz,
            #                                      bias_need=False).cuda()
            self.encoder: nn.Module = myencoder(hps=hp).cuda()
            self.decoder: nn.Module = DecoderRNN().cuda()
        else:
            self.encoder: nn.Module = EncoderGCN(hp.graph_number, hp.graph_picture_size, hp.out_f_num, hp.Nz,
                                                 bias_need=False)
            self.decoder: nn.Module = DecoderRNN()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr, betas=(0.5,0.999))
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr, betas=(0.5,0.999))
        self.eta_step = hp.eta_min
        
        # 添加训练历史记录
        self.train_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'rpcl_loss': [],
            'alpha_loss': [],
            'gaussian_loss': [],
            'mse_loss': [],
            'lr': [],
            'epoch': []
        }
        
        self.val_history = {
            'total_loss': [],
            'reconstruction_loss': [],
            'rpcl_loss': [],
            'alpha_loss': [],
            'gaussian_loss': [],
            'mse_loss': []
        }

    def validate(self, val_dataset, epoch):
        """在验证集上计算损失，不更新模型参数"""
        self.encoder.eval()
        self.decoder.eval()
        
        total_loss_list = []
        reconstruction_loss_list = []
        rpcl_loss_list = []
        alpha_loss_list = []
        gaussian_loss_list = []
        mse_loss_list = []
        
        # 验证几个batch（减少计算量）
        val_batches = min(10, len(val_dataset.sketches) // hp.batch_size)
        if val_batches == 0:
            val_batches = 1  # 至少验证一个batch
        
        with torch.no_grad():
            for i in range(val_batches):
                batch, lengths, graphs, adjs = val_dataset.make_batch(hp.batch_size)
                
                if hp.use_cuda:
                    batch = batch.cuda()
                    graphs = graphs.cuda()
                    adjs = adjs.cuda()
                    # lengths 是列表，不需要移动到CUDA
                
                # encode
                z, mu, sigma, mseloss, rpclloss, update_data = self.encoder(graphs)
                
                # decode
                if hp.use_cuda:
                    sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * hp.batch_size).cuda().unsqueeze(0)
                else:
                    sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * hp.batch_size).unsqueeze(0)
                batch_init = torch.cat([sos, batch], 0)
                z_stack = torch.stack([z] * (hp.Nmax + 1))
                inputs = torch.cat([batch_init, z_stack], 2)
                
                pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, _, _ = self.decoder(inputs, z)
                
                # 计算损失
                mask, dx, dy, p = self.make_target(batch, lengths)
                LR = self.reconstruction_loss(mask, dx, dy, p, epoch)
                
                # 计算总损失（注意：验证时不更新eta_step和assign）
                eta_step_val = 1.0 - 0.999 * (0.9999 ** (max(0, epoch - hp.br))) if epoch >= hp.br else 0
                val_loss = LR + rpclloss[0] * eta_step_val + 0.5 * mseloss
                
                # 记录各项损失
                total_loss_list.append(val_loss.item())
                reconstruction_loss_list.append(LR.item())
                rpcl_loss_list.append(rpclloss[0].item())
                alpha_loss_list.append(rpclloss[1].item())
                gaussian_loss_list.append(rpclloss[2].item())
                mse_loss_list.append(mseloss.item())
        
        # 返回平均值
        return {
            'total_loss': np.mean(total_loss_list),
            'reconstruction_loss': np.mean(reconstruction_loss_list),
            'rpcl_loss': np.mean(rpcl_loss_list),
            'alpha_loss': np.mean(alpha_loss_list),
            'gaussian_loss': np.mean(gaussian_loss_list),
            'mse_loss': np.mean(mse_loss_list)
        }

    def save_training_history(self):
        """保存训练历史到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        history_file = os.path.join(hp.model_save, f'training_history_{timestamp}.json')
        
        history_data = {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'hyperparameters': {
                'lr': hp.lr,
                'batch_size': hp.batch_size,
                'Nz': hp.Nz,
                'br': hp.br,
                'eta_min': hp.eta_min
            }
        }
        
        with open(history_file, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        print(f"训练历史已保存到: {history_file}")

    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.train_history['epoch']:  # 如果没有训练数据，不绘图
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training and Validation Curves', fontsize=16)
        
        # 1. Total Loss
        axes[0,0].plot(self.train_history['epoch'], self.train_history['total_loss'], 'b-', label='Train', linewidth=2)
        if self.val_history['total_loss']:
            val_epochs = [e for e in self.train_history['epoch'] if e % 50 == 0]  # 假设每100个epoch验证一次
            if len(val_epochs) == len(self.val_history['total_loss']):
                axes[0,0].plot(val_epochs, self.val_history['total_loss'], 'r-', label='Val', linewidth=2)
        axes[0,0].set_title('Total Loss')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Reconstruction Loss
        axes[0,1].plot(self.train_history['epoch'], self.train_history['reconstruction_loss'], 'b-', label='Train', linewidth=2)
        if self.val_history['reconstruction_loss']:
            if len(val_epochs) == len(self.val_history['reconstruction_loss']):
                axes[0,1].plot(val_epochs, self.val_history['reconstruction_loss'], 'r-', label='Val', linewidth=2)
        axes[0,1].set_title('Reconstruction Loss')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. RPCL Loss
        axes[1,0].plot(self.train_history['epoch'], self.train_history['rpcl_loss'], 'b-', label='Train', linewidth=2)
        if self.val_history['rpcl_loss']:
            if len(val_epochs) == len(self.val_history['rpcl_loss']):
                axes[1,0].plot(val_epochs, self.val_history['rpcl_loss'], 'r-', label='Val', linewidth=2)
        axes[1,0].set_title('RPCL Loss')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. MSE Loss
        axes[1,1].plot(self.train_history['epoch'], self.train_history['mse_loss'], 'b-', label='Train', linewidth=2)
        if self.val_history['mse_loss']:
            if len(val_epochs) == len(self.val_history['mse_loss']):
                axes[1,1].plot(val_epochs, self.val_history['mse_loss'], 'r-', label='Val', linewidth=2)
        axes[1,1].set_title('MSE Loss')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Loss')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = os.path.join(hp.model_save, f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存到: {plot_file}")
        plt.show()

    def lr_decay(self, optimizer: optim):
        """Decay learning rate by a factor of lr_decay"""
        for param_group in optimizer.param_groups:
            if param_group['lr'] > hp.min_lr:
                param_group["lr"] = (0.001-0.00001)*(0.9999**(self.encoder.global_.item()/3.))+0.00001
               # param_group['lr'] *= hp.lr_decay
        return optimizer

    def make_target(self, batch, lengths):
        """
        batch torch.Size([129, 100, 5])  Nmax batch_size
        """
        if hp.use_cuda:
            eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).cuda().unsqueeze(
                0)  # torch.Size([1, 100, 5])
        else:
            eos = torch.stack([torch.Tensor([0, 0, 0, 0, 1])] * batch.size()[1]).unsqueeze(0)  # max of len(strokes)

        batch = torch.cat([batch, eos], 0)
        mask = torch.zeros(hp.Nmax + 1, batch.size()[1])
        for indice, length in enumerate(lengths):  # len(lengths) = batchsize
            mask[:length, indice] = 1
        if hp.use_cuda:
            mask = mask.cuda()
        dx = torch.stack([batch.data[:, :, 0]] * hp.M, 2)  # torch.Size([130, 100, 20])
        dy = torch.stack([batch.data[:, :, 1]] * hp.M, 2)  # torch.Size([130, 100, 20])
        p1 = batch.data[:, :, 2]  # torch.Size([130, 100])
        p2 = batch.data[:, :, 3]
        p3 = batch.data[:, :, 4]
        p = torch.stack([p1, p2, p3], 2)  # torch.Size([130, 100, 3])
        return mask, dx, dy, p

    def train(self, epoch, val_dataset=None):
        self.encoder.train()
        self.decoder.train()

        self.encoder.global_.data = self.encoder.global_.data+1

        batch, lengths, graphs, adjs = sketch_dataset.make_batch(hp.batch_size)
        #print(batch, lengths)

        # encode:
        # z, self.mu, self.sigma = self.encoder(batch, hp.batch_size)  # in here, Z is sampled from N(mu, sigma)
        #z, self.mu, self.sigma, _ = self.encoder(graphs, adjs)  # in here, Z is sampled from N(mu, sigma)
        z, self.mu, self.sigma, mseloss, rpclloss, update_data = self.encoder(graphs)

        # torch.Size([100, 128]) torch.Size([100, 128]) torch.Size([100, 128])
        # print(z.shape, self.mu.shape, self.sigma.shape)
  
        # create start of sequence:
        if hp.use_cuda:
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * hp.batch_size).cuda().unsqueeze(0)
            # torch.Size([1, 100, 5])
        else:
            sos = torch.stack([torch.Tensor([0, 0, 1, 0, 0])] * hp.batch_size).unsqueeze(0)
        # had sos at the begining of the batch:
        batch_init = torch.cat([sos, batch], 0)  # torch.Size([130, 100, 5])
        # expend z to be ready to concatenate with inputs:
        z_stack = torch.stack([z] * (hp.Nmax + 1))  # torch.Size([130, 100, 128])
        # inputs is concatenation of z and batch_inputs
        inputs = torch.cat([batch_init, z_stack], 2)  # torch.Size([130, 100, 133])

        # decode:
        self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, self.rho_xy, self.q, _, _= self.decoder(inputs, z)

        # prepare targets:
        mask, dx, dy, p = self.make_target(batch, lengths)
        # prepare optimizers:
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # update eta for LKL:
        if epoch>=hp.br:
            self.eta_step = 1.0 - 0.999 * (0.9999 ** (epoch-hp.br))
            self.encoder.assign(update_data)
        else:
            self.eta_step = 0 
        #self.eta_step = np.clip(self.eta_step,0,0.25)
        # self.eta_step = 1 - (1 - hp.eta_min) * hp.R
        #self.eta_step = 0.1-0.095*(0.9999**epoch)
        # compute losses:
        # LKL = self.kullback_leibler_loss()
        LR = self.reconstruction_loss(mask, dx, dy, p, epoch)
        # loss = LR + LKL
        if LR [LR!=LR].size(0)>0:
            print("NAN LR")
            return
        loss = LR + rpclloss[0]*self.eta_step + 0.5*mseloss
        # gradient step
        loss.backward()  # all torch.Tensor has backward.
        # gradient cliping
        nn.utils.clip_grad_norm_(self.encoder.parameters(), hp.grad_clip)
        nn.utils.clip_grad_norm_(self.decoder.parameters(), hp.grad_clip)
        # optim step
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # some print and save:
        if epoch % 1 == 0:
            # print('epoch', epoch, 'loss', loss.item(), 'LR', LR.item(), 'LKL', LKL.item())
            print('gcn, epoch -> ', epoch, 'loss', loss.item(), 'LR', LR.item(),'mse:',mseloss.item()
                  ,' rpcl', rpclloss[0].item(),"alpha_loss",rpclloss[1].item(),"gaussian_loss",rpclloss[2].item(), "eta_step", self.eta_step)
            
            # 记录训练历史
            self.train_history['epoch'].append(epoch)
            self.train_history['total_loss'].append(loss.item())
            self.train_history['reconstruction_loss'].append(LR.item())
            self.train_history['rpcl_loss'].append(rpclloss[0].item())
            self.train_history['alpha_loss'].append(rpclloss[1].item())
            self.train_history['gaussian_loss'].append(rpclloss[2].item())
            self.train_history['mse_loss'].append(mseloss.item())
            self.train_history['lr'].append(self.encoder_optimizer.param_groups[0]['lr'])
            
            self.encoder_optimizer = self.lr_decay(self.encoder_optimizer)  # modify optimizer after one step.
            self.decoder_optimizer = self.lr_decay(self.decoder_optimizer)
        
        # 验证过程：每100个epoch验证一次
        if epoch % 50 == 0 and epoch > 0 and val_dataset is not None:
            print(f"\n--- 在验证集上评估 epoch {epoch} ---")
            val_losses = self.validate(val_dataset, epoch)
            
            # 记录验证历史
            self.val_history['total_loss'].append(val_losses['total_loss'])
            self.val_history['reconstruction_loss'].append(val_losses['reconstruction_loss'])
            self.val_history['rpcl_loss'].append(val_losses['rpcl_loss'])
            self.val_history['alpha_loss'].append(val_losses['alpha_loss'])
            self.val_history['gaussian_loss'].append(val_losses['gaussian_loss'])
            self.val_history['mse_loss'].append(val_losses['mse_loss'])
            
            print(f"验证集 - Total Loss: {val_losses['total_loss']:.4f}, "
                  f"Reconstruction Loss: {val_losses['reconstruction_loss']:.4f}, "
                  f"RPCL Loss: {val_losses['rpcl_loss']:.4f}, "
                  f"MSE Loss: {val_losses['mse_loss']:.4f}")
            print("--- 验证完成 ---\n")
            
            # 切换回训练模式
            self.encoder.train()
            self.decoder.train()
        
        if epoch == 0:
            return
        if epoch % 500 == 0:
            self.conditional_generation(epoch)
        if epoch % 1000 == 0:
            self.save(epoch)
            # 保存GMM参数
            self.save_gmm_parameters(epoch)
            # 定期保存训练历史和绘图
            if epoch > 0:
                self.save_training_history()
                self.plot_training_curves()

    def bivariate_normal_pdf(self, dx, dy):
        z_x = ((dx - self.mu_x) / self.sigma_x) ** 2
        z_y = ((dy - self.mu_y) / self.sigma_y) ** 2
        z_xy = (dx - self.mu_x) * (dy - self.mu_y) / (self.sigma_x * self.sigma_y)
        z = z_x + z_y - 2 * self.rho_xy * z_xy
        exp = torch.exp(-z / (2 * (1 - self.rho_xy ** 2)))
        norm = 2 * np.pi * self.sigma_x * self.sigma_y * torch.sqrt(1 - self.rho_xy ** 2)
        return exp / norm

    def reconstruction_loss(self, mask, dx, dy, p, epoch):
        pdf = self.bivariate_normal_pdf(dx, dy)  # torch.Size([130, 100, 20])
        # stroke
        LS = -torch.sum(mask * torch.log(1e-3 + torch.sum(self.pi * pdf, 2))) / float(hp.batch_size)
        # position
        LP = -torch.sum(p * torch.log(1e-3 + self.q)) / float(hp.batch_size)
        return LS + LP

    def kullback_leibler_loss(self):
        LKL = -0.5 * torch.sum(1 + self.sigma - self.mu ** 2 - torch.exp(self.sigma)) \
              / float(hp.Nz * hp.batch_size)
        if hp.use_cuda:
            KL_min = torch.Tensor([hp.KL_min]).cuda().detach()
        else:
            KL_min = torch.Tensor([hp.KL_min]).detach()
        return hp.wKL * self.eta_step * torch.max(LKL, KL_min)

    def save(self, epoch):
        # sel = np.random.rand()
        torch.save(self.encoder.state_dict(), \
                   f'{hp.model_save}/encoderRNN_epoch_{epoch}.pth')
        torch.save(self.decoder.state_dict(), \
                   f'{hp.model_save}/decoderRNN_epoch_{epoch}.pth')

    def save_gmm_parameters(self, epoch):
        """
        保存encoder中所有GMM参数（共k组GMM）
        
        保存的参数包括：
        - de_mu: GMM均值参数，形状 (k, z_size)
        - de_sigma2: GMM方差参数，形状 (k, z_size)  
        - de_alpha: GMM混合权重，形状 (k, 1)
        - k: GMM组件数量
        - z_size: 隐空间维度
        """
        import numpy as np
        import os
        
        print(f"\n{'='*60}")
        print(f"保存GMM参数 - epoch {epoch}")
        print(f"{'='*60}")
        
        # 获取GMM参数
        k = self.encoder.k  # GMM组件数量
        z_size = self.encoder.z_size  # 隐空间维度
        
        # 提取参数数据
        de_mu = self.encoder.de_mu.data.cpu().numpy()  # (k, z_size)
        de_sigma2 = self.encoder.de_sigma2.data.cpu().numpy()  # (k, z_size)
        de_alpha = self.encoder.de_alpha.data.cpu().numpy()  # (k, 1)
        
        print(f"GMM组件数量 (k): {k}")
        print(f"隐空间维度 (z_size): {z_size}")
        print(f"de_mu 形状: {de_mu.shape}")
        print(f"de_sigma2 形状: {de_sigma2.shape}")
        print(f"de_alpha 形状: {de_alpha.shape}")
        
        # 创建保存目录
        gmm_save_dir = f'{hp.model_save}/gmm_parameters'
        os.makedirs(gmm_save_dir, exist_ok=True)
        
        # 保存为NPZ格式（包含所有GMM参数）
        gmm_file = f'{gmm_save_dir}/gmm_parameters_epoch_{epoch}.npz'
        np.savez(gmm_file, 
                 de_mu=de_mu,
                 de_sigma2=de_sigma2,
                 de_alpha=de_alpha,
                 k=k,
                 z_size=z_size,
                 epoch=epoch)
        
        print(f"✓ GMM参数已保存到: {gmm_file}")
        
        # 额外保存为单独的numpy文件以便分析
        # 保存每个GMM组件的参数
        for i in range(k):
            component_file = f'{gmm_save_dir}/gmm_component_{i}_epoch_{epoch}.npz'
            np.savez(component_file,
                     mu=de_mu[i],  # (z_size,)
                     sigma2=de_sigma2[i],  # (z_size,)
                     alpha=de_alpha[i, 0],  # scalar
                     component_id=i,
                     epoch=epoch)
        
        print(f"✓ 各GMM组件参数已分别保存到: {gmm_save_dir}/gmm_component_*_epoch_{epoch}.npz")
        
        # 保存GMM参数的统计信息
        stats = {
            'mu_mean': float(np.mean(de_mu)),
            'mu_std': float(np.std(de_mu)),
            'mu_min': float(np.min(de_mu)),
            'mu_max': float(np.max(de_mu)),
            'sigma2_mean': float(np.mean(de_sigma2)),
            'sigma2_std': float(np.std(de_sigma2)),
            'sigma2_min': float(np.min(de_sigma2)),
            'sigma2_max': float(np.max(de_sigma2)),
            'alpha_mean': float(np.mean(de_alpha)),
            'alpha_std': float(np.std(de_alpha)),
            'alpha_min': float(np.min(de_alpha)),
            'alpha_max': float(np.max(de_alpha)),
            'alpha_sum': float(np.sum(de_alpha)),
            'dominant_components': [int(x) for x in np.argsort(de_alpha.flatten())[-5:][::-1]],  # top 5 components by weight
            'epoch': epoch,
            'k': k,
            'z_size': z_size
        }
        
        stats_file = f'{gmm_save_dir}/gmm_statistics_epoch_{epoch}.json'
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✓ GMM统计信息已保存到: {stats_file}")
        
        # 打印一些关键统计信息
        print(f"\nGMM参数统计:")
        print(f"  mu范围: [{stats['mu_min']:.4f}, {stats['mu_max']:.4f}], 均值: {stats['mu_mean']:.4f}, 标准差: {stats['mu_std']:.4f}")
        print(f"  sigma2范围: [{stats['sigma2_min']:.4f}, {stats['sigma2_max']:.4f}], 均值: {stats['sigma2_mean']:.4f}, 标准差: {stats['sigma2_std']:.4f}")
        print(f"  alpha范围: [{stats['alpha_min']:.4f}, {stats['alpha_max']:.4f}], 均值: {stats['alpha_mean']:.4f}, 标准差: {stats['alpha_std']:.4f}")
        print(f"  alpha总和: {stats['alpha_sum']:.4f}")
        print(f"  权重最高的5个组件: {stats['dominant_components']}")
        
        print(f"{'='*60}")

    def load(self, encoder_name, decoder_name):
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)

    def conditional_generation(self, epoch):
        batch, lengths, graphs, adjs = sketch_dataset.make_batch(1)
        # should remove dropouts:
        self.encoder.train(False)
        self.decoder.train(False)
        # encode:
        z, _, _, _, _, _ = self.encoder(graphs)
        if hp.use_cuda:
            sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda()
        else:
            sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
        s = sos
        seq_x = []
        seq_y = []
        seq_z = []
        hidden_cell = None
        for i in range(hp.Nmax):
            input = torch.cat([s, z.unsqueeze(0)], 2)  # start of stroke concatenate with z
            # decode:
            self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
            self.rho_xy, self.q, hidden, cell= \
                self.decoder(input, z, hidden_cell)
            hidden_cell = (hidden, cell)
            # sample from parameters:
            s, dx, dy, pen_down, eos = self.sample_next_state()
            # ------
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(pen_down)
            if eos:
                print(i)
                break
        # visualize result:
        x_sample = np.cumsum(seq_x, 0)
        y_sample = np.cumsum(seq_y, 0) 
        z_sample = np.array(seq_z)
        sequence = np.stack([x_sample, y_sample, z_sample]).T
        make_image(sequence, epoch)

    def sample_next_state(self):
        """
        softmax
        """

        def adjust_temp(pi_pdf):
            pi_pdf = np.log(1e-3 + np.abs(pi_pdf)) / hp.temperature
            # pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= (pi_pdf.sum())
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0, 0, :].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(hp.M, p=pi)
        # get pen state:
        q = self.q.data[0, 0, :].cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p=q)
        # get mixture params:
        mu_x = self.mu_x.data[0, 0, pi_idx]
        mu_y = self.mu_y.data[0, 0, pi_idx]
        sigma_x = self.sigma_x.data[0, 0, pi_idx]
        sigma_y = self.sigma_y.data[0, 0, pi_idx]
        rho_xy = self.rho_xy.data[0, 0, pi_idx]
        x, y = sample_bivariate_normal(mu_x, mu_y, sigma_x, sigma_y, rho_xy, greedy=False)  # get samples.
        next_state = torch.zeros(5)
        next_state[0] = x
        next_state[1] = y
        next_state[q_idx + 2] = 1
        if hp.use_cuda:
            return next_state.cuda().view(1, 1, -1), x, y, q_idx == 1, q_idx == 2
        else:
            return next_state.view(1, 1, -1), x, y, q_idx == 1, q_idx == 2


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":
    model = Model()
    print(get_parameter_number(model.encoder))
    print(get_parameter_number(model.decoder))
    epoch_load =0
    if epoch_load !=0:
        model.load(f'{hp.model_save}/encoderRNN_epoch_{epoch_load}.pth',
                f'{hp.model_save}/decoderRNN_epoch_{epoch_load}.pth')
    model.encoder_optimizer = optim.Adam(model.encoder.parameters(), hp.lr)
    model.decoder_optimizer = optim.Adam(model.decoder.parameters(), hp.lr)
    
    # 创建验证集数据加载器
    print("正在加载验证集...")
    val_dataset = SketchesDataset(hp.data_location, hp.category, "valid")
    print(f"验证集加载完成，共{len(val_dataset.sketches)}个样本")
       
    for epoch in range(52001):
        if epoch <= epoch_load:  
            model.lr_decay(model.encoder_optimizer)
            model.lr_decay(model.decoder_optimizer)
            continue
        #if epoch_load:
            #model.load(f'./{hp.model_save}/encoderRNN_epoch_{epoch_load}.pth',
                       #f'./{hp.model_save}/deco  derRNN_epoch_{epoch_load}.pth')
        #print(model.encoder.de_alpha)
        model.train(epoch, val_dataset)
    
    # 训练完成后保存最终的历史记录和绘图
    print("\n训练完成，正在保存训练历史和绘图...")
    model.save_training_history()
    model.plot_training_curves()
    print("所有任务完成！")

    '''
                                           
    model.conditional_generation(0)
    #'''
