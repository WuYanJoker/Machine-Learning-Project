#!/usr/bin/env python
# complete_sketch.py
"""
草图补全对比可视化
功能：从数据集中抽取完整草图 → 删除后x% → encoder编码 → decoder重建 → 三图对比

命令行：
    python complete_sketch.py airplane 0.3
"""
import os
import sys
import numpy as np
import cv2
import torch
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sequence_sketch import load_model
from utils.inference_sketch_processing import draw_three
from utils.sketch_processing import make_graph
from Pix2Seq import hp


def load_and_truncate_sketch(data_root, category, trunc_ratio=0.3, sample_idx=0):
    """加载并截断草图序列"""
    npz_path = os.path.join(data_root, f"{category}.npz")
    data = np.load(npz_path, allow_pickle=True, encoding='latin1')
    key = 'z' if 'z' in data else 'train'
    sketches = data[key]
    
    original = sketches[sample_idx]
    total_len = len(original)
    truncate_point = int(total_len * (1 - trunc_ratio))
    truncated = original[:truncate_point].copy()
    
    print(f"  原始序列长度: {total_len}, 截断后: {len(truncated)}")
    return original, truncated, truncate_point

def encode_partial_sketch(encoder, partial_sketch):
    """编码部分草图（先转换为图表示）"""
    # 1. 将 stroke3 序列转换为图
    graph_tensor, adj_matrix = make_graph(partial_sketch, 
                                           graph_num=hp.graph_number,
                                           graph_picture_size=hp.graph_picture_size,
                                           mask_prob=0.0)
    
    # 2. 准备 batch 数据
    # graph_tensor shape: (graph_num, 3, graph_picture_size, graph_picture_size)
    # 转换为 torch tensor 并添加 batch 维度
    graphs = torch.tensor(graph_tensor, dtype=torch.float32).unsqueeze(0)  # (1, graph_num, 3, H, W)
    
    # 3. 移动到与模型相同的设备
    device = next(encoder.parameters()).device
    graphs = graphs.to(device)
    
    # 4. 编码 (encoder 返回6个值)
    with torch.no_grad():
        z, mu, sigma, mse_loss, rpcl_loss, q_params = encoder(graphs)
        latent_z = z.squeeze(0).cpu().numpy()  # (128,)
    
    print(f"  隐变量 shape: {latent_z.shape}")
    return latent_z

def decode_and_render(pix2seq_model, latent_z, save_path=None):
    """从隐变量解码并渲染（使用完整Pix2Seq模型）"""
    from utils.inference_sketch_processing import draw_three
    import torch.nn.functional as F
    
    # 准备隐变量
    z_tensor = torch.tensor(latent_z, dtype=torch.float32).unsqueeze(0)  # (1, 128)
    device = next(pix2seq_model.decoder.parameters()).device
    z_tensor = z_tensor.to(device)
    
    # 生成草图序列（使用Pix2Seq的采样逻辑）
    max_len = 100  # 最大生成长度
    sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).to(device)
    
    with torch.no_grad():
        # 初始化隐藏状态
        hidden, cell = torch.split(torch.tanh(pix2seq_model.decoder.fc_hc(z_tensor)), 
                                   hp.dec_hidden_size, 1)
        hidden_cell = (hidden.unsqueeze(0).contiguous(), cell.unsqueeze(0).contiguous())
        
    #     # 生成序列
    #     generated_sequence = []
    #     current_input = sos
        
    #     for i in range(max_len):
    #         # 解码器前向传播
    #         pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell = \
    #             pix2seq_model.decoder(current_input, z_tensor, hidden_cell)
            
    #         # 从混合高斯分布采样下一个点
    #         # 简化采样：选择概率最高的混合成分
    #         idx = torch.argmax(pi.squeeze(), dim=-1)
            
    #         # 获取采样点的坐标
    #         x = mu_x.squeeze()[idx]
    #         y = mu_y.squeeze()[idx]
            
    #         # 获取笔状态
    #         pen_idx = torch.argmax(q.squeeze(), dim=-1)
    #         pen_state = [1, 0, 0] if pen_idx == 0 else ([0, 1, 0] if pen_idx == 1 else [0, 0, 1])
            
    #         # 构建stroke (dx, dy, pen_state)
    #         stroke = np.array([x.cpu().numpy(), y.cpu().numpy(), pen_idx.cpu().numpy()])
    #         generated_sequence.append(stroke)
            
    #         # 检查是否结束
    #         if pen_idx == 2:  # 结束标志
    #             break
            
    #         # 准备下一个输入
    #         current_input = torch.tensor([[stroke]], dtype=torch.float32).to(device)
    #         hidden_cell = (hidden, cell)
    
    # # 转换为numpy数组
    # generated_sequence = np.array(generated_sequence)

    generated_sequence = pix2seq_model.conditional_generate_by_z(z_tensor)
    
    # 渲染为图像
    img = draw_three(generated_sequence, img_size=256)
    
    if save_path:
        cv2.imwrite(save_path, img)
    
    return img

def render_three_views(original, truncated, reconstructed, category, trunc_ratio):
    """渲染三图对比视图"""
    # 渲染三幅图
    img_orig = draw_three(original, img_size=256)
    
    img_trunc = draw_three(truncated, img_size=256)
    h, w = img_trunc.shape[:2]
    cv2.line(img_trunc, (0, h//2), (w, h//2), (0, 0, 255), 2)
    cv2.putText(img_trunc, f"Trunc {trunc_ratio:.0%}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    img_recon = reconstructed
    
    # 拼接三图
    comparison = np.hstack([img_orig, img_trunc, img_recon])
    
    # 添加标题
    title_bar = np.ones((60, comparison.shape[1], 3), dtype=np.uint8) * 255
    cv2.putText(title_bar, f"{category} | Completion (trunc={trunc_ratio:.0%})", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # 添加标签
    h_text = img_orig.shape[0] + 80
    cv2.putText(title_bar, "Original", (10, h_text), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(title_bar, "Truncated Input", (img_orig.shape[1] + 10, h_text), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(title_bar, "Decoder Reconstruction", (img_orig.shape[1]*2 + 10, h_text), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    comparison = np.vstack([title_bar, comparison])
    return comparison

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('category', type=str, help='草图类别')
    parser.add_argument('trunc_ratio', type=float, help='截断比例 (0-1)')
    parser.add_argument('--sample_idx', type=int, default=0, help='样本索引')
    parser.add_argument('--save_dir', type=str, default='./completion_results', help='保存目录')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"草图补全: {args.category} | 截断比例: {args.trunc_ratio:.0%}")
    print(f"{'='*60}")
    
    # 1. 加载并截断
    print("\n【Step 1】加载并截断草图...")
    data_root = "1024Rpcl/dataset"
    original, truncated, _ = load_and_truncate_sketch(
        data_root, args.category, args.trunc_ratio, args.sample_idx
    )
    
    # 2. 编码
    print("\n【Step 2】加载encoder并编码...")
    model = load_model(dir_path = "/home/sda/jbx2/CS3308/1024Rpcl/model_save_ek20_1227/")
    encoder = model.encoder
    latent_z = encode_partial_sketch(encoder, truncated)
    
    # 3. 解码
    print("\n【Step 3】加载decoder并重建...")
    # 设置评估模式（分别设置encoder和decoder）
    model.encoder.eval()
    model.decoder.eval()
    
    reconstructed = decode_and_render(
        model, latent_z,
        os.path.join(args.save_dir, f"{args.category}_recon.png")
    )
    
    # 4. 渲染对比
    print("\n【Step 4】渲染对比图...")
    comparison = render_three_views(original, truncated, reconstructed, 
                                    args.category, args.trunc_ratio)
    
    # 5. 保存
    out_path = os.path.join(args.save_dir, f"{args.category}_compare.png")
    cv2.imwrite(out_path, comparison)
    print(f"\n✅ 对比图已保存 → {out_path}")
    
    # 6. 保存序列数据
    np.save(os.path.join(args.save_dir, f"{args.category}_orig.npy"), original)
    np.save(os.path.join(args.save_dir, f"{args.category}_trunc.npy"), truncated)
    print(f"   原始序列: {args.save_dir}/{args.category}_orig.npy")
    print(f"   截断序列: {args.save_dir}/{args.category}_trunc.npy")

if __name__ == "__main__":
    main()
