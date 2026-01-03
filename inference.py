import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from hyper_params import hp
import numpy as np
import matplotlib.pyplot as plt
import PIL

import torch
import torch.nn as nn
from torch import optim
from encoder import myencoder
from decoder import DecoderRNN
from utils.inference_sketch_processing import make_graph, draw_three, make_graph_


################################# load and prepare data
class SketchesDataset:
    def __init__(self, path: str, category: list, mode="train"):
        self.sketches = None
        self.sketches_categroy_count: list = []
        self.sketches_normed = None
        """上面两个sketches 是完全拷贝的"""
        self.max_sketches_len = 0
        self.path = path
        self.category = category
        self.mode = mode

        tmp_sketches = []
        for c in self.category:
            dataset = np.load(os.path.join(self.path, c), encoding='latin1', allow_pickle=True)
            tmp_sketches.append(dataset[self.mode])
            self.sketches_categroy_count.append(len(dataset[self.mode]))
            print(f"dataset: {c} added.")
        data_sketches = np.concatenate(tmp_sketches)
        print(f"length of train set: {len(data_sketches)}")

        data_sketches = self.purify(data_sketches)  # data clean.  # remove too long and too stort sketches.
        self.sketches = data_sketches.copy()
        self.sketches_normed = self.normalize(data_sketches)
        self.Nmax = self.max_size(data_sketches)  # max size of a sketch.
        print(f"max length of sketch is: {self.Nmax}")

    def max_size(self, sketches):
        """返回所有sketch中 转折最多的一个sketch"""
        sizes = [len(sketch) for sketch in sketches]
        return max(sizes)

    def purify(self, sketches):
        """
        移除太短或过长的stroke
        移除单个stroke 太长的."""
        data = []
        for sketch in sketches:
            if hp.max_seq_length >= sketch.shape[0] > hp.min_seq_length:  # remove small and too long sketches.
                sketch = np.minimum(sketch, 1000)  # remove large gaps.
                sketch = np.maximum(sketch, -1000)
                sketch = np.array(sketch, dtype=np.float32)  # change it into float32
                data.append(sketch)
        return data

    @staticmethod
    def calculate_normalizing_scale_factor(sketches):
        """计算所有sketches中的标准差"""
        data = []
        for sketch in sketches:
            for stroke in sketch:
                data.append(stroke)
        return np.std(np.array(data))

    def normalize(self, sketches):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor.
        将所有的sketches 标准化, 即除以标准差. 使得方差等于1"""
        data = []
        #scale_factor = self.calculate_normalizing_scale_factor(sketches)
        scale_factor = 44.18034
        for sketch in sketches:
            sketch[:, 0:2] /= scale_factor
            data.append(sketch)
        return data

    def get_sample(self, sketch_index: int):
        """
        :return:
        返回 batch, lengths. batch为sketches的连接, lengths是每一个sketch的长度列表
        """
        batch_idx = [sketch_index]
        batch_sketches = [self.sketches_normed[idx] for idx in batch_idx]  # 从标准化后的抽取
        batch_sketches_graphs = [self.sketches[idx] for idx in batch_idx]  # 图卷积使用, 图卷积不能使用归一化后的
        sketches = []
        lengths = []
        graphs = []  # (batch_size * graphs_num_constant, x, y) # 注意按照 graphs num 切分
        adjs = []
        index = 0
        for _sketch in batch_sketches:
            len_seq = len(_sketch[:, 0])  # sketch 笔画数量
            new_sketch = np.zeros((self.Nmax, 5))  # new a _sketch, all length of sketch in size is Nmax.
            new_sketch[:len_seq, :2] = _sketch[:, :2]  # 1. 将x y拷贝进新的sketch

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


def sample_bivariate_normal(mu_x: torch.Tensor, mu_y: torch.Tensor,
                            sigma_x: torch.Tensor, sigma_y: torch.Tensor,
                            rho_xy: torch.Tensor, greedy=False):
    """
    根据网络输出, 进行采样
    1. 获取 x, y的均值及标准差
    2. 计算相关系数
    """
    mu_x = mu_x.item()
    mu_y = mu_y.item()
    sigma_x = sigma_x.item()
    sigma_y = sigma_y.item()
    rho_xy = rho_xy.item()
    # inputs must be floats
    if greedy:
        return mu_x, mu_y
    mean = [mu_x, mu_y]

    sigma_x *= np.sqrt(hp.temperature)  # 乘以热度开根号
    sigma_y *= np.sqrt(hp.temperature)

    cov = [[sigma_x * sigma_x, rho_xy * sigma_x * sigma_y],
           [rho_xy * sigma_x * sigma_y, sigma_y * sigma_y]]
    x = np.random.multivariate_normal(mean, cov, 1)
    return x[0][0], x[0][1]


def make_image(sequence, sketch_index, name='_output_', path="./visualize/"):
    """分离strokes, 并画图"""
    strokes = np.split(sequence, np.where(sequence[:, 2] > 0)[0] + 1)  # 指出所有满足条件的坐标, +1 是因为split类似于[m:n]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for s in strokes:
        plt.plot(s[:, 0], -s[:, 1])
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(),
                                    canvas.tostring_rgb())
    os.makedirs(f"{path}/{name}", exist_ok=True)
    name = f"{path}" + str(sketch_index) + name + '.jpg'
    pil_image.save(name, "JPEG")
    plt.close("all")


"""
encoder and decoder modules
"""


class Model:
    def __init__(self):
        if hp.use_cuda:
            self.encoder: nn.Module = myencoder(hps = hp).cuda()
            self.decoder: nn.Module = DecoderRNN().cuda()
        else:
            self.encoder: nn.Module = myencoder(hps =hp)
            self.decoder: nn.Module = DecoderRNN()
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), hp.lr)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), hp.lr)
        self.eta_step = hp.eta_min

        self.pi: torch.Tensor = torch.Tensor()
        self.z: torch.Tensor = torch.Tensor()
        self.mu_x: torch.Tensor = torch.Tensor()
        self.mu_y: torch.Tensor = torch.Tensor()
        self.sigma_x: torch.Tensor = torch.Tensor()
        self.sigma_y: torch.Tensor = torch.Tensor()
        self.rho_xy: torch.Tensor = torch.Tensor()
        self.q: torch.Tensor = torch.Tensor()

    def validate(self, sketch_dataset, save_middle_path="visualize"):
        self.encoder.eval()
        self.decoder.eval()
        # some print and save:
        with torch.no_grad():
            self.conditional_generation(sketch_dataset, save_middle_path)

    def validate_simple(self, sketch_dataset, save_middle_path="validate_simple", samples_per_category=20):
        """
        简化版验证函数 - 每类只测试指定数量的样例
        
        Args:
            sketch_dataset: 数据集对象
            save_middle_path: 保存路径
            samples_per_category: 每类测试样本数，默认20
        """
        self.encoder.eval()
        self.decoder.eval()
        
        print(f"开始简化验证 - 每类测试{samples_per_category}个样例")
        
        # 用于保存所有类别的结果
        all_result_z_list = []
        all_ret_z_list = []
        
        # 遍历每个类别
        for cat_idx, category_file in enumerate(sketch_dataset.category):

            generated_sketches = []  # 收集所有生成的草图序列

            category_name = category_file.split(".")[0]
            category_count = sketch_dataset.sketches_categroy_count[cat_idx]
            
            print(f"\n=== 处理类别: {category_name} ===")
            
            # 获取该类别在数据集中的起始和结束索引
            if cat_idx == 0:
                start_idx = 0
            else:
                start_idx = sum(sketch_dataset.sketches_categroy_count[:cat_idx])
            end_idx = start_idx + sketch_dataset.sketches_categroy_count[cat_idx]
            
            # 随机选择指定数量的样本索引（或全部如果少于指定数量）
            total_samples = end_idx - start_idx
            sample_count = min(samples_per_category, total_samples)
            
            if total_samples <= samples_per_category:
                # 如果样本数少于要求，测试全部样本
                test_indices = [np.int64(i) for i in range(start_idx, end_idx)]
                # print(test_indices)
            else:
                # 随机选择指定数量的样本
                # import numpy as np 
                test_indices = np.random.choice(range(start_idx, end_idx), 
                                              size=sample_count, 
                                              replace=False)
                # print(test_indices)
            
            print(f"类别{category_name}共有{total_samples}个样本，测试{len(test_indices)}个")
            
            # 存储该类别的结果
            result_z_list = []
            ret_z_list = []
            
            # 测试选定的样本
            for idx, sketch_index in enumerate(test_indices):
                # print(sketch_index, type(sketch_index))
                if idx % 5 == 0:  # 每5个样本打印一次进度
                    print(f"  进度: {idx+1}/{len(test_indices)}")
                
                try:
                    # 1. 获取样本数据
                    batch, lengths, graphs, adjs = sketch_dataset.get_sample(sketch_index)
                    
                    # 2. 编码原始草图 → 隐向量（使用mu）
                    self.z, mu, sigma, _, _, _ = self.encoder(graphs)
                    result_z_list.append(mu.detach().cpu().numpy())
                    
                    # 3. 从隐向量生成新草图
                    if hp.use_cuda:
                        sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda()
                    else:
                        sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
                    
                    s = sos
                    seq_x = []
                    seq_y = []
                    seq_z = []
                    hidden_cell = None
                    
                    # 自回归生成
                    for i in range(hp.Nmax):
                        _input = torch.cat([s, mu.unsqueeze(0)], 2)
                        self.pi, self.mu_x, self.mu_y, \
                        self.sigma_x, self.sigma_y, \
                        self.rho_xy, self.q, hidden, cell = \
                            self.decoder(_input, mu, hidden_cell)
                        
                        hidden_cell = (hidden, cell)
                        s, dx, dy, pen_down, eos = self.sample_next_state()
                        
                        seq_x.append(dx)
                        seq_y.append(dy)
                        seq_z.append(pen_down)
                        
                        if eos:
                            break
                    
                    # 4. 重新编码生成的草图
                    ret_seq_x = np.asarray(seq_x)
                    ret_seq_y = np.asarray(seq_y)
                    ret_seq_z = np.asarray(seq_z)
                    ret_sequence = np.stack([ret_seq_x, ret_seq_y, ret_seq_z]).T
                    
                    # 将生成的序列转换为图像并重新编码
                    _graph_tensor, _adj_matrix = make_graph_(ret_sequence, 
                                                           graph_num=hp.graph_number,
                                                           graph_picture_size=hp.graph_picture_size, 
                                                           mask_prob=0.0)
                    
                    if hp.use_cuda:
                        _, ret_mu, _, _, _, _ = self.encoder(_graph_tensor.cuda().unsqueeze(0))
                    else:
                        _, ret_mu, _, _, _, _ = self.encoder(_graph_tensor.unsqueeze(0))
                    
                    ret_z_list.append(ret_mu.detach().cpu().numpy())
                    
                    # 5. 可视化生成结果（只可视化前5个，节省空间）
                    if idx < 20:  # 每类只可视化前5个结果
                        x_sample = np.cumsum(seq_x, 0)
                        y_sample = np.cumsum(seq_y, 0)
                        z_sample = np.array(seq_z)
                        sequence = np.stack([x_sample, y_sample, z_sample]).T
                        
                        # 生成重建图像
                        try:
                            _sketch = np.stack([seq_x, seq_y, seq_z]).T
                            generated_sketches.append(_sketch)
                            reconstructed_cv = draw_three(_sketch, img_size=256)

                            # # 生成渐变成完整图像的图片序列
                            # if idx < 5:
                            #     # reconstructed_sequence = []
                            #     # print(len(seq_x))
                            #     for i in range(1, len(seq_x)):
                            #         sx = seq_x[0: i + 1]
                            #         sy = seq_y[0: i + 1]
                            #         sz = seq_z[0: i + 1]
                            #         sk = np.stack([sx, sy, sz]).T
                            #         # print(sk.size())
                            #         cv = draw_three(sk, img_size=256)
                            #         # reconstructed_sequence.append(cv)
                            #         # 渲染成图片
                                    
                            #         if cv is None or cv.size == 0:
                            #             print(f"  警告: 第{i+1}个渲染失败")
                            #             continue
                                    
                            #         # 保存图片
                            #         filename = f"sketch_{category_name}_{idx}_{i:03d}.png"
                            #         save_path = os.path.join(save_middle_path, filename)
                            #         cv2.imwrite(save_path, cv)

                            
                            # 获取原始草图进行对比
                            try:
                                original_seq = sketch_dataset.sketches_normed[sketch_index]
                                original_cv = draw_three(original_seq, img_size=256)
                                
                                # 保存重建后的草图
                                recon_save_dir = f"{save_middle_path}/reconstructed/{category_name}"
                                os.makedirs(recon_save_dir, exist_ok=True)
                                recon_path = f"{recon_save_dir}/{idx:03d}.png"
                                cv2.imwrite(recon_path, reconstructed_cv)
                                print(f"  重建草图已保存: {recon_path}")
                                
                                # 创建并排对比图
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                                
                                # 原始图像
                                ax1.imshow(original_cv)
                                ax1.set_title(f"Original #{sketch_index}", fontsize=10)
                                ax1.axis('off')
                                
                                # 重建图像
                                ax2.imshow(reconstructed_cv)
                                ax2.set_title(f"Reconstructed #{sketch_index}", fontsize=10)
                                ax2.axis('off')
                                
                                plt.suptitle(f"{category_name.title()} - Sample {sketch_index}", fontsize=12)
                                plt.tight_layout()
                                # 确保保存目录存在
                                comparison_dir = f"{save_middle_path}/comparison"
                                os.makedirs(comparison_dir, exist_ok=True)
                                plt.savefig(f"{comparison_dir}/{category_name}_comparison_{idx:03d}.png", dpi=150, bbox_inches='tight')
                                plt.close()
                                
                                print(f"  Saved comparison: {category_name}_comparison_{idx:03d}.png")
                                
                            except Exception as e:
                                print(f"  Error loading original sketch: {e}")
                                # 如果无法获取原图，只保存重建图
                                save_dir = f"{save_middle_path}/sketch/{category_name}"
                                os.makedirs(save_dir, exist_ok=True)
                                cv2.imwrite(f"{save_dir}/sample_{idx:03d}.jpg", reconstructed_cv)
                                print(f"  Saved reconstructed image only: sample_{idx:03d}.jpg")
                                
                        except Exception as e:
                            print(f"  Visualization error: {str(e)}")
                
                except Exception as e:
                    print(f"  处理样本{sketch_index}时出错: {str(e)}")
                    continue
            
            # 保存该类别的结果
            all_result_z_list.extend(result_z_list)
            all_ret_z_list.extend(ret_z_list)
            
            np.savez(os.path.join(save_middle_path, f'{category_name}_sketches.npz'), sketches=np.array(generated_sketches, dtype=object))

            # 保存隐向量数据
            if len(result_z_list) > 0:
                os.makedirs(f"{save_middle_path}/npz", exist_ok=True)
                os.makedirs(f"{save_middle_path}/retnpz", exist_ok=True)
                
                np.savez(f"{save_middle_path}/npz/{category_name}.npz", 
                        z=np.array(result_z_list))
                np.savez(f"{save_middle_path}/retnpz/{category_name}.npz", 
                        z=np.array(ret_z_list))
            
            print(f"  类别{category_name}处理完成，共处理{len(result_z_list)}个样本")
        
        print(f"\n=== 简化验证完成 ===")
        print(f"总计处理了{len(sketch_dataset.category)}个类别")
        print(f"每类测试{samples_per_category}个样本")
        print(f"结果保存在: {save_middle_path}")
        
        
        
        return all_result_z_list, all_ret_z_list

    def conditional_generation(self, sketch_dataset, save_middle_path="visualize"):
        count = 0
        category_flag = 0
        category_name = sketch_dataset.category[category_flag].split(".")[0]
        category_count = sketch_dataset.sketches_categroy_count[category_flag]

        result_z_list = []
        ret_z_list = []

        for sketch_index, sketch in enumerate(sketch_dataset.sketches_normed):
            print(sketch_index)
            batch, lengths, graphs, adjs = sketch_dataset.get_sample(sketch_index)
            # encode:
            self.z, mu, sigma, _, _, _ = self.encoder(graphs)
            # result_z_list.append(self.z.cpu().numpy())
            result_z_list.append(mu.cpu().numpy())

            count += 1

            # if sketch_index % 100 != 0 or True:
            #     continue
            print(f"drawing {category_name} {count}")
            if hp.use_cuda:
                sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda()
            else:
                sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
            s = sos
            seq_x = []
            seq_y = []
            seq_z = []
            hidden_cell = None
            for i in range(hp.Nmax):  # Nmax = 151
                _input = torch.cat([s, mu.unsqueeze(0)], 2)  # start of stroke concatenate with z
                # decode:

                self.pi, self.mu_x, self.mu_y, self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = \
                    self.decoder(_input, mu, hidden_cell)

                hidden_cell = (hidden, cell)
                # sample from parameters:
                s, dx, dy, pen_down, eos = self.sample_next_state()
                # ------
                seq_x.append(dx)
                seq_y.append(dy)
                seq_z.append(pen_down)
                if eos:
                    # print(i)
                    break
            ret_seq_x = np.asarray(seq_x)
            ret_seq_y = np.asarray(seq_y)
            ret_seq_z = np.asarray(seq_z)
            ret_sequence = np.stack([ret_seq_x, ret_seq_y, ret_seq_z]).T
            _graph_tensor, _adj_matrix = make_graph_(ret_sequence, graph_num=hp.graph_number,
                                                    graph_picture_size=hp.graph_picture_size, mask_prob=0.0)

            _, ret_mu, _, _, _, _ = self.encoder(_graph_tensor.cuda().unsqueeze(0))
            ret_z_list.append(ret_mu.cpu().numpy())


            # visualize result:
            x_sample = np.cumsum(seq_x, 0)  # 累加, 梯形求和
            y_sample = np.cumsum(seq_y, 0)
            z_sample = np.array(seq_z)
            sequence = np.stack([x_sample, y_sample, z_sample]).T
            # # visualize result:
            _sketch = np.stack([seq_x, seq_y, seq_z]).T
            try:
                sketch_cv = draw_three(_sketch, img_size=256)
            except Exception as e:
                print('draw error')
                _sketch = np.zeros((256, 256, 1))

            print('draw:')
            os.makedirs(f"{save_middle_path}/sketch/{category_name}", exist_ok=True)
            cv2.imwrite(f"{save_middle_path}/sketch/{category_name}/{sketch_index}.jpg", sketch_cv)
            # cv2.imwrite(f"{save_middle_path}/sketch/{category_name}/{sketch_index}.jpg", _sketch)

            #make_image(sequence, count - 1, name=f"_{category_name}", path=f"./{save_middle_path}/sketch/")
            if count == category_count:
                os.makedirs(f"{save_middle_path}/npz", exist_ok=True)
                np.savez(f"./{save_middle_path}/npz/{category_name}.npz", z=np.array(result_z_list))
                result_z_list = []
                os.makedirs(f"{save_middle_path}/retnpz", exist_ok=True)
                np.savez(f"./{save_middle_path}/retnpz/{category_name}.npz", z=np.array(ret_z_list))
                ret_z_list = []
                count = 0
                print(f"{category_name} finished")
                category_flag += 1
                if category_flag < len(hp.category):
                    category_name = sketch_dataset.category[category_flag].split(".")[0]
                    category_count = sketch_dataset.sketches_categroy_count[category_flag]


                                                                      
                      

    def save_z(self, sketch_dataset, save_middle_path="batch_z_results", batch_size = 50):
        """
        批量化计算并保存result_z和ret_z
        对所有样本进行编码和重建，比validate_simple更快
        
        Args:
            sketch_dataset: 数据集
            save_middle_path: 保存路径
        """
        print(f"\n{'='*60}")
        print("批量化保存隐空间向量")
        print(f"{'='*60}")
        
        self.encoder.eval()
        self.decoder.eval()
        
        batch_size = batch_size  # 批处理大小
        total_samples = len(sketch_dataset.sketches)
        
        print(f"总样本数: {total_samples}")
        print(f"批处理大小: {batch_size}")
        
        with torch.no_grad():
            # 遍历每个类别
            for cat_idx, category_file in enumerate(sketch_dataset.category):
                category_name = category_file.split(".")[0]
                print(f"\n处理类别: {category_name}")
                
                # 获取该类别的样本范围
                if cat_idx == 0:
                    start_idx = 0
                else:
                    start_idx = sum(sketch_dataset.sketches_categroy_count[:cat_idx])
                end_idx = start_idx + sketch_dataset.sketches_categroy_count[cat_idx]
                
                category_count = sketch_dataset.sketches_categroy_count[cat_idx]
                print(f"  样本范围: {start_idx} - {end_idx-1} (共{category_count}个)")
                
                # 存储结果
                result_z_list = []  # 原始z
                ret_z_list = []     # 重建z
                
                # 批处理处理该类别所有样本
                for i in range(0, category_count, batch_size):
                    batch_end = min(i + batch_size, category_count)
                    batch_indices = list(range(start_idx + i, start_idx + batch_end))
                    
                    if i % 100 == 0:
                        print(f"  进度: {i}/{category_count}")
                    
                    try:
                        # 批量获取数据
                        batch_data = []
                        lengths_list = []
                        graphs_list = []
                        adjs_list = []
                        
                        for idx in batch_indices:
                            batch, lengths, graphs, adjs = sketch_dataset.get_sample(idx)
                            batch_data.append(batch)
                            lengths_list.append(lengths)
                            graphs_list.append(graphs)
                            adjs_list.append(adjs)
                        
                        # 拼接批次数据
                        if hp.use_cuda:
                            batch_tensor = torch.cat(batch_data, dim=1).cuda()  # (Nmax, batch_size, 5)
                            graphs_tensor = torch.cat(graphs_list, dim=0).cuda()  # (batch_size, ...)
                        else:
                            batch_tensor = torch.cat(batch_data, dim=1)  # (Nmax, batch_size, 5)
                            graphs_tensor = torch.cat(graphs_list, dim=0)  # (batch_size, ...)
                        
                        # 批量编码
                        z_batch, mu_batch, sigma_batch, _, _, _ = self.encoder(graphs_tensor)
                        
                        # 存储原始z (使用mu而不是z，与validate_simple一致)
                        result_z_list.extend(mu_batch.cpu().numpy())
                        
                        # 批量生成和重编码
                        batch_ret_z = []
                        for j in range(mu_batch.shape[0]):
                            # 生成新草图
                            z_single = mu_batch[j:j+1]  # (1, 128)
                            if hp.use_cuda:
                                z_single = z_single.cuda()
                            
                            # 生成序列
                            seq_x, seq_y, seq_z = [], [], []
                            hidden_cell = None
                            sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
                            if hp.use_cuda:
                                sos = sos.cuda()
                            s = sos
                            
                            for step in range(hp.Nmax):
                                _input = torch.cat([s, z_single.unsqueeze(0)], 2)
                                self.pi, self.mu_x, self.mu_y, \
                                self.sigma_x, self.sigma_y, \
                                self.rho_xy, self.q, hidden, cell = \
                                    self.decoder(_input, z_single, hidden_cell)
                                hidden_cell = (hidden, cell)
                                s, dx, dy, pen_down, eos = self.sample_next_state()
                                
                                seq_x.append(dx)
                                seq_y.append(dy)
                                seq_z.append(pen_down)
                                if eos:
                                    break
                            
                            # 重建序列
                            ret_seq_x = np.asarray(seq_x)
                            ret_seq_y = np.asarray(seq_y)
                            ret_seq_z = np.asarray(seq_z)
                            ret_sequence = np.stack([ret_seq_x, ret_seq_y, ret_seq_z]).T
                            
                            # 重编码
                            _graph_tensor, _ = make_graph_(ret_sequence, 
                                                         graph_num=hp.graph_number,
                                                         graph_picture_size=hp.graph_picture_size,
                                                         mask_prob=0.0)
                            
                            if hp.use_cuda:
                                _, ret_mu, _, _, _, _ = self.encoder(_graph_tensor.cuda().unsqueeze(0))
                            else:
                                _, ret_mu, _, _, _, _ = self.encoder(_graph_tensor.unsqueeze(0))
                            
                            batch_ret_z.append(ret_mu.cpu().numpy())
                        
                        ret_z_list.extend(batch_ret_z)
                        
                    except Exception as e:
                        print(f"  批处理 {i}-{batch_end-1} 时出错: {e}")
                        continue
                
                # 保存该类别的结果
                print(f"  保存类别 {category_name} 结果...")
                os.makedirs(f"{save_middle_path}/npz", exist_ok=True)
                os.makedirs(f"{save_middle_path}/retnpz", exist_ok=True)
                
                # 保存原始z
                result_z_array = np.array(result_z_list)
                np.savez(f"{save_middle_path}/npz/{category_name}.npz", z=result_z_array)
                print(f"  ✓ 保存原始z: {save_middle_path}/npz/{category_name}.npz (形状: {result_z_array.shape})")
                
                # 保存重建z
                ret_z_array = np.array(ret_z_list)
                np.savez(f"{save_middle_path}/retnpz/{category_name}.npz", z=ret_z_array)
                print(f"  ✓ 保存重建z: {save_middle_path}/retnpz/{category_name}.npz (形状: {ret_z_array.shape})")
                
                print(f"  类别 {category_name} 处理完成")
        
        print(f"\n{'='*60}")
        print("批量化保存完成！")
        print(f"{'='*60}")


    def conditional_generate_by_z(self, z, index=-1, plt_show=False):  #
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            if hp.use_cuda:
                sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).cuda()
            else:
                sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
            s = sos
            seq_x = []
            seq_y = []
            seq_z = []
            hidden_cell = None
            for i in range(151):  # Nmax = 151
                _input = torch.cat([s, z.unsqueeze(0)], 2)  # start of stroke concatenate with z
                # decode:
                self.pi, \
                self.mu_x, self.mu_y, \
                self.sigma_x, self.sigma_y, \
                self.rho_xy, self.q, hidden, cell = self.decoder(_input, z, hidden_cell)
                hidden_cell = (hidden, cell)
                # sample from parameters:
                s, dx, dy, pen_down, eos = self.sample_next_state()
                # ------
                seq_x.append(dx)
                seq_y.append(dy)
                seq_z.append(pen_down)
                if eos:
                    seq_x.append(0)
                    seq_y.append(0)
                    seq_z.append(True)
                    break
            # visualize result:
            x_sample = np.cumsum(seq_x, 0)  # 累加, 梯形求和
            y_sample = np.cumsum(seq_y, 0)
            z_sample = np.array(seq_z)
                                      
            sequence = np.stack([x_sample, y_sample, z_sample]).T
            if plt_show:
                make_image(sequence, index, name=f"_z_generated", path="./visualize/generate_z/")
            _sketch = np.stack([seq_x, seq_y, seq_z]).T
            return _sketch

    def sample_next_state(self):
        def adjust_temp(pi_pdf):
            """
            SoftMax
            """
            pi_pdf = np.log(pi_pdf) / hp.temperature
            pi_pdf -= pi_pdf.max()
            pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()
            return pi_pdf

        # get mixture indice:
        pi = self.pi.data[0, 0, :].cpu().numpy()
        pi = adjust_temp(pi)
        pi_idx = np.random.choice(hp.M, p=pi)  # 抽一个数字
        # get pen state:
        q = self.q.data[0, 0, :].cpu().numpy()
        q = adjust_temp(q)
        q_idx = np.random.choice(3, p=q)  # 抽一个数字
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

    def load(self, encoder_name, decoder_name):
        saved_encoder = torch.load(encoder_name)
        saved_decoder = torch.load(decoder_name)
        self.encoder.load_state_dict(saved_encoder)
        self.decoder.load_state_dict(saved_decoder)


if __name__ == "__main__":
    import random
    import glob
    import cv2

    hp.mask_prob = 0.0
    sketch_dataset = SketchesDataset(hp.data_location, hp.category, "test")
    hp.Nmax = sketch_dataset.Nmax
    # hp.Nmax = 177
    hp.temperature = 0.01
    # hp.Nmax = 151 for v2_1
    # hp.Nmax = 177 for masked
    model = Model()
    # model.encoder.cuda()
    # model.decoder.cuda()
    # model.load("./model_save_v2_1/encoderRNN_epoch_99000.pth",
    #            "./model_save_v2_1/decoderRNN_epoch_99000.pth")
    #model.load(f"./{hp.model_save}/encoderRNN_epoch_8000_sgy.pth",
    #           f"./{hp.model_save}/decoderRNN_epoch_8000_sgy.pth")
    model.load(f"{hp.model_save}/encoderRNN_epoch_52000.pth",
               f"{hp.model_save}/decoderRNN_epoch_52000.pth")

    print(hp.mask_prob, hp.Nmax)
    
    # 选择运行模式
    mode = "validate_simple"  # 可以是 "validate_simple" 或 "batch_z"
    # test_samples = 20  # 测试样本数量
    
    if mode == "validate_simple":
        # 原始逐个样本处理（慢）
        print("运行原始validate_simple模式...")
        model.validate_simple(sketch_dataset,
                           save_middle_path="validate_simple_results",
                           samples_per_category=30)  # 少量样本测试
    elif mode == "batch_z":
        # 新的批量化处理（快）
        print("运行新的批量化save_z模式...")
        # 只处理少量样本进行测试
        # hp.category = ["airplane.npz"]  # 只用1个类别测试
        test_dataset = SketchesDataset(hp.data_location, hp.category, "test")
        model.save_z(test_dataset, save_middle_path="batch_z_results", batch_size=100)
    
    exit(0)
                         
                                       
''' 
    # generate images by z or mu
    root_path = f"result/visualize2/146000/{hp.mask_prob}"
    for each_npz_path in glob.glob(f"./{root_path}/npz/*.npz"):
        _npz = np.load(each_npz_path, allow_pickle=True, encoding="latin1")["z"]
        npz_path = each_npz_path.split("/")[-1]
        cate_name = npz_path.replace(".npz", "")
        if os.path.exists(f"./{root_path}/images/{cate_name}"):
            pass
        else:
            os.makedirs(f"./{root_path}/images/{cate_name}")
        for index, each_vector in enumerate(_npz):
            _sketch = model.conditional_generate_by_z(torch.Tensor(each_vector).cuda())
            sketch_image_cv = draw_three(_sketch, show=False, img_size=256)
            cv2.imwrite(f"./{root_path}/images/{cate_name}/{index}.jpg", sketch_image_cv)
            print(f"{cate_name} {index} finished")
        print(f"{cate_name} finished")
    exit(0)
'''
