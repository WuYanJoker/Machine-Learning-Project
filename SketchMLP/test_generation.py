import os
import sys
import pickle
import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from Dataset import GenerationDataset
from Utils import save_checkpoint, load_checkpoint
from Networks5 import net
from Hyper_params import hp
from tensorboardX import SummaryWriter
import torch.optim as optim
import random
from tqdm.auto import tqdm
from Utils import get_cosine_schedule_with_warmup
seed = 1010
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"
print("***********- ***********- READ DATA and processing-*************")

# Dataloader for generated sketches under Data/generation
gen_dataset = GenerationDataset()

# 如果没有成功读取到任何生成样本，则直接退出，避免后续仍然构建模型导致段错误。
if len(gen_dataset) == 0:
    print("[test_generation] GenerationDataset 中样本数为 0，无法进行识别。")
    print("[test_generation] 请确认 npz 文件包含 key 'sketches'，且其中保存为纯数值数组，"
          "而不是旧环境的 pickled object；重新保存后再运行本脚本。")
    sys.exit(0)

dataloader_Gen = DataLoader(gen_dataset, batch_size=hp.batchsize, shuffle=False,
                            num_workers=int(hp.nThreads))

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

# log_dir = hp.log
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# log_path = os.path.join(log_dir, 'eval_{}_ckpts.txt'.format(hp.model_name))

print('load pretrain model')
model_name = 'MoE_epoch_40'
# if hp.Dataset=='QuickDraw':
#     model_name = 'QD'
# elif hp.Dataset == 'QuickDraw414k':
#     model_name = 'QD414k'

checkpoint = torch.load('./pretrain/'+model_name+'.pkl')['net_state_dict']
model.load_state_dict(checkpoint)


def load_label_map(txt_path="Data/QuickDraw414k/picture_files/tiny_train_set.txt"):
    """Build mapping from label index to category name using training list."""
    label_to_name = {}
    if not os.path.exists(txt_path):
        return label_to_name
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            rel_path, label_str = parts
            try:
                label = int(label_str)
            except ValueError:
                continue
            class_name = rel_path.split('/')[0]
            if label not in label_to_name:
                label_to_name[label] = class_name
    return label_to_name


class trainer:
    def __init__(self, model, label_map):
        self.model = model
        self.label_map = label_map

    def predict_generation(self, loader):
        self.model.eval()
        loader = tqdm(loader)
        all_results = []

        print("\n************Generation Data Recognition*************")
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                imgs = batch['sketch_img']
                seqs = batch['sketch_points']

                if len(hp.gpus) > 0:
                    imgs = imgs.cuda()
                    seqs = seqs.cuda()

                logits, img_logsoftmax, seq_logsoftmax, cv_important = self.model(imgs, seqs)
                probs = torch.softmax(logits, dim=1)
                top_probs, top_indices = probs.topk(5, dim=1)

                for i in range(logits.size(0)):
                    sample_idx = batch.get('sample_idx', None)
                    if sample_idx is not None:
                        sample_index_val = int(sample_idx[i].item())
                    else:
                        sample_index_val = batch_idx * hp.batchsize + i

                    source_category = batch.get('source_category', None)
                    if source_category is not None:
                        # source_category is a list of strings
                        src_cat = source_category[i]
                    else:
                        src_cat = ''

                    preds = []
                    for rank in range(top_indices.size(1)):
                        cls_idx = int(top_indices[i, rank].item())
                        cls_name = self.label_map.get(cls_idx, 'unknown')
                        prob = float(top_probs[i, rank].item())
                        preds.append((cls_idx, cls_name, prob))

                    all_results.append({
                        'sample_index': sample_index_val,
                        'source_category': src_cat,
                        'predictions': preds,
                    })

        return all_results




print('''***********- Evaluating -*************''')
params_total = sum(p.numel() for p in model.parameters())
print("Number of parameter: %.2fM"%(params_total/1e6))
label_map = load_label_map()
Trainer = trainer(model, label_map)
results = Trainer.predict_generation(dataloader_Gen)

# Print recognition results to stdout
for res in results:
    print("\nSample {} (source category: {})".format(res['sample_index'], res['source_category']))
    for rank, (cls_idx, cls_name, prob) in enumerate(res['predictions'], start=1):
        print("  Top{}: {} (id={}) prob={:.4f}".format(rank, cls_name, cls_idx, prob))

# Save detailed results for later analysis
# 1) Pickle 文件，便于 Python 直接加载
# with open('generation_results.pkl', 'wb') as f:
#     pickle.dump(results, f)

# 2) 文本/CSV 文件，便于人直接查看或用 Excel 处理
txt_path = 'generation_results.txt'
with open(txt_path, 'w', encoding='utf-8') as f_txt:
    f_txt.write('sample_index\tsource_category\trank\tcls_idx\tcls_name\tprob\n')
    for res in results:
        sample_index = res['sample_index']
        source_category = res['source_category']
        for rank, (cls_idx, cls_name, prob) in enumerate(res['predictions'], start=1):
            line = f"{sample_index}\t{source_category}\t{rank}\t{cls_idx}\t{cls_name}\t{prob:.6f}\n"
            f_txt.write(line)

print(f"\n[test_generation] 结果已保存到 generation_results.pkl 和 {txt_path}")
