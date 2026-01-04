import os
import pickle
import time
from Dataset import get_dataloader
import torch
import numpy as np
import torch.nn as nn
from Utils import save_checkpoint, load_checkpoint
from Networks5 import net
from Hyper_params import hp
from tensorboardX import SummaryWriter
import torch.optim as optim
import random
from metrics import AverageMeter,accuracy
from tqdm.auto import tqdm
from Utils import get_cosine_schedule_with_warmup
seed = 1010
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"
print("***********- ***********- READ DATA and processing-*************")
dataloader_Train, dataloader_Test, dataloader_Valid = get_dataloader()

print("***********- loading model -*************")
if(len(hp.gpus)==0):#cpu
    model = net()
elif(len(hp.gpus)==1):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(hp.gpus[0])
    model = net().cuda()
else:#multi gpus
    gpus = ','.join(str(i) for i in hp.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    model = net().cuda()
    gpus = [i for i in range(len(hp.gpus))]
    model = torch.nn.DataParallel(model, device_ids=gpus)


# 评估 ckpts-tiny 中所有模型，因此不再加载 pretrain 权重
ckpt_dir = hp.model_save  # 例如 'ckpts-tiny'
ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pkl')])
print('found {} checkpoints in {}'.format(len(ckpt_files), ckpt_dir))

log_dir = hp.log
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, 'eval_{}_ckpts.txt'.format(hp.model_name))

# print('load pretrain model')
# if hp.Dataset=='QuickDraw':
#     model_name = 'QD'
# elif hp.Dataset == 'QuickDraw414k':
#     model_name = 'QD414k'

# checkpoint = torch.load('./pretrain/'+model_name+'.pkl')['net_state_dict']
# model.load_state_dict(checkpoint)
loss_f = nn.CrossEntropyLoss(label_smoothing=0.1)

class weight_record():
    def __init__(self):
        self.weight_img_record = np.zeros([hp.categories])
        self.cat_num = np.zeros([hp.categories])

    def update(self, labels, weight):
        for i,label in enumerate(labels):
            self.weight_img_record[label] = self.weight_img_record[label]+weight[i][0]
            self.cat_num[label] = self.cat_num[label] + 1

    def calculate(self):
        self.weight_img_record = self.weight_img_record / self.cat_num

class trainer:
    def __init__(self, loss_f, model):
        self.loss_f = loss_f
        self.model = model
        self.iter = 0


    def valid_epoch(self, loader, epoch):
        self.model.eval()
        loader = tqdm(loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        print("\n************Evaluation*************")
        for batch_idx, (batch) in enumerate(loader):
            with torch.no_grad():
                imgs = batch['sketch_img']
                seqs = batch['sketch_points']
                labels = batch['sketch_label']
                time_total = 0
                if (len(hp.gpus) > 0):
                    batch_imgs, batch_seqs, batch_labels = imgs.cuda(), seqs.cuda(), labels.cuda()
                predicted, img_logsoftmax, seq_logsoftmax, cv_important = self.model(batch_imgs, batch_seqs)
                w_r.update(labels, cv_important)
                loss, mix_loss, img_loss, seq_loss = self.myloss(predicted, img_logsoftmax, seq_logsoftmax, batch_labels)
                loss = loss
                loss = loss.detach().cpu().numpy()
                mix_loss = mix_loss.detach().cpu().numpy()
                img_loss = img_loss.detach().cpu().numpy()
                seq_loss = seq_loss.detach().cpu().numpy()
                losses.update(loss.item(), batch_imgs.size(0))

                err1, err5 = accuracy(predicted.data, batch_labels, topk=(1, 5))
                top1.update(err1.item(), batch_imgs.size(0))
                top5.update(err5.item(), batch_imgs.size(0))

        return top1.avg, top5.avg, losses.avg

    def myloss(self,predicted, img_ls, seq_ls, labels):
        # print(predicted.size(),labels.size())#[128, 10]) torch.Size([128])
        mix_loss = self.loss_f(predicted, labels)
        img_loss = self.loss_f(img_ls, labels)
        seq_loss = self.loss_f(seq_ls, labels)
        loss = hp.mix_weight * mix_loss + hp.img_weight * img_loss + hp.seq_weight * seq_loss
        return loss, mix_loss, img_loss, seq_loss

    def run(self, train_loder, val_loder, test_loder):
        test_err1, test_err5, test_loss = self.valid_epoch(test_loder, 0)
        w_r.calculate()
        print(w_r.weight_img_record)
        with open(model_name+'.pkl', 'wb') as f:
            pickle.dump(w_r, f)
        print("\nval_loss:{:.4f} | err1:{:.4f} | err5:{:.4f}".format(test_loss, test_err1, test_err5))




print('''***********- Evaluating -*************''')
params_total = sum(p.numel() for p in model.parameters())
print("Number of parameter: %.2fM"%(params_total/1e6))
Trainer = trainer(loss_f, model)
# Trainer.run(dataloader_Train, dataloader_Valid, dataloader_Test)

# 依次评估 ckpt_dir 中的所有 checkpoint，并将结果写入日志文件
with open(log_path, 'a') as lf:
    lf.write('\n===== Eval start: {} =====\n'.format(time.strftime('%Y-%m-%d %H:%M:%S')))
    lf.write('ckpt_name\ttest_loss\terr1\terr5\n')

    for ckpt_name in ckpt_files:
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        print('\n[Eval] loading checkpoint:', ckpt_path)
        state = torch.load(ckpt_path)
        if isinstance(state, dict) and 'net_state_dict' in state:
            model.load_state_dict(state['net_state_dict'])
        else:
            model.load_state_dict(state)

        # 每个模型单独统计一遍权重记录
        w_r = weight_record()

        test_err1, test_err5, test_loss = Trainer.valid_epoch(dataloader_Test, 0)
        w_r.calculate()

        log_line = '{}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(ckpt_name, test_loss, test_err1, test_err5)
        print('[Eval Result]', log_line.strip())
        lf.write(log_line)

    lf.write('===== Eval end =====\n')
