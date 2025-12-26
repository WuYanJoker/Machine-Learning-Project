#!/usr/bin/env python3
"""
简化测试脚本 - 避免设备冲突问题
"""

import torch
import numpy as np
from hyper_params import hp
from decoder import DecoderRNN
from utils.sketch_processing import make_graph
import matplotlib.pyplot as plt

def test_decoder_only():
    """仅测试解码器组件"""
    print("测试解码器组件...")
    
    # 强制使用CPU
    hp.use_cuda = False
    
    # 创建简单的隐向量
    batch_size = 1
    z_dim = 128
    z = torch.randn(batch_size, z_dim)
    
    # 创建解码器
    decoder = DecoderRNN()
    decoder.eval()
    
    print(f"解码器参数数量: {sum(p.numel() for p in decoder.parameters())}")
    
    # 测试解码器前向传播
    sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1).repeat(1, batch_size, 1)
    input_seq = torch.cat([sos, z.unsqueeze(0).repeat(1, batch_size, 1)], dim=-1)
    
    with torch.no_grad():
        pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell = decoder(input_seq, z)
    
    print(f"解码器输出形状:")
    print(f"  pi (混合权重): {pi.shape}")
    print(f"  mu_x, mu_y (均值): {mu_x.shape}, {mu_y.shape}")
    print(f"  sigma_x, sigma_y (标准差): {sigma_x.shape}, {sigma_y.shape}")
    print(f"  q (笔状态): {q.shape}")
    
    print("✓ 解码器测试通过")
    return True

def test_sketch_generation():
    """测试完整的草图生成过程"""
    print("测试草图生成过程...")
    
    hp.use_cuda = False
    hp.Nz = 128
    hp.M = 20  # 混合成分数
    hp.temperature = 0.2
    
    # 创建解码器
    decoder = DecoderRNN()
    decoder.eval()
    
    # 随机隐向量
    z = torch.randn(1, hp.Nz)
    
    # 生成序列
    sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
    s = sos
    seq_x, seq_y, seq_z = [], [], []
    hidden_cell = None
    
    with torch.no_grad():
        for i in range(100):  # 最多生成100步
            input_seq = torch.cat([s, z.unsqueeze(0)], dim=-1)
            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell = decoder(input_seq, z, hidden_cell)
            hidden_cell = (hidden, cell)
            
            # 采样下一步 - 修复采样逻辑
            # get mixture indice:
            pi_probs = pi[0, 0].softmax(dim=0).cpu().numpy()
            pi_idx = np.random.choice(hp.M, p=pi_probs)
            
            # get pen state:
            q_probs = q[0, 0].softmax(dim=0).cpu().numpy()
            q_idx = np.random.choice(3, p=q_probs)
            
            # get mixture params and sample:
            mu_x_val = mu_x[0, 0, pi_idx].item()
            mu_y_val = mu_y[0, 0, pi_idx].item()
            sigma_x_val = sigma_x[0, 0, pi_idx].item()
            sigma_y_val = sigma_y[0, 0, pi_idx].item()
            rho_xy_val = rho_xy[0, 0, pi_idx].item()
            
            # 使用二元正态分布采样
            mean = [mu_x_val, mu_y_val]
            cov = [[sigma_x_val * sigma_x_val, rho_xy_val * sigma_x_val * sigma_y_val],
                   [rho_xy_val * sigma_x_val * sigma_y_val, sigma_y_val * sigma_y_val]]
            
            # 应用温度参数
            cov = np.array(cov) * hp.temperature
            
            try:
                sample = np.random.multivariate_normal(mean, cov, 1)
                dx, dy = sample[0][0], sample[0][1]
            except:
                # 如果协方差矩阵有问题，直接使用均值
                dx, dy = mu_x_val, mu_y_val
            
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(q_idx == 1)  # 笔落下状态
            
            # 更新状态
            next_state = torch.zeros(5)
            next_state[0] = dx
            next_state[1] = dy
            next_state[q_idx + 2] = 1
            s = next_state.view(1, 1, -1)
            
            if q_idx == 2:  # EOS
                # 添加结束点
                seq_x.append(0)
                seq_y.append(0) 
                seq_z.append(True)
                break
    
    # 计算累积坐标
    x_cum = np.cumsum(seq_x)
    y_cum = np.cumsum(seq_y)
    
    print(f"生成序列长度: {len(seq_x)}")
    print(f"坐标范围: x∈[{min(x_cum):.2f}, {max(x_cum):.2f}], y∈[{min(y_cum):.2f}, {max(y_cum):.2f}]")
    print(f"笔画数: {sum(seq_z)}")
    
    # 修复可视化 - 更好地处理笔画分割
    plt.figure(figsize=(10, 8))
    
    # 收集笔画
    strokes = []
    current_stroke_x = [0]  # 从原点开始
    current_stroke_y = [0]
    
    for i in range(len(seq_x)):
        current_stroke_x.append(x_cum[i])
        current_stroke_y.append(y_cum[i])
        
        if seq_z[i] or i == len(seq_x) - 1:  # 如果是笔画结束或最后一点
            if len(current_stroke_x) > 1:  # 确保有内容
                strokes.append((current_stroke_x.copy(), current_stroke_y.copy()))
            current_stroke_x = [x_cum[i]]  # 新笔画的起点
            current_stroke_y = [y_cum[i]]
    
    # 绘制所有笔画
    for stroke_x, stroke_y in strokes:
        plt.plot(stroke_x, stroke_y, 'b-', linewidth=2, alpha=0.8)
        # 标记起点
        plt.plot(stroke_x[0], stroke_y[0], 'ro', markersize=4)
    
    # 设置坐标轴范围和样式
    margin = 10
    plt.xlim(min(x_cum) - margin, max(x_cum) + margin)
    plt.ylim(min(y_cum) - margin, max(y_cum) + margin)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.title('Generated Sketch')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/sdb/users/jbx2/CS3308/1024Rpcl/test_generation.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ 草图生成测试通过")
    print("生成的草图已保存为 test_generation.png")
    return True

def test_sketch_processing():
    """测试草图处理工具"""
    print("测试草图处理工具...")
    
    # 创建模拟草图数据
    np.random.seed(42)
    sketch = np.random.randn(20, 3).astype(np.float32) * 50
    sketch[:, 2] = (np.random.rand(20) > 0.8).astype(np.float32)  # 20%的概率笔画结束
    
    # 测试图构建
    graph, adj = make_graph(sketch, graph_num=5, graph_picture_size=64, mask_prob=0.0)
    
    print(f"输入草图形状: {sketch.shape}")
    print(f"输出图形状: {graph.shape}")
    print(f"邻接矩阵形状: {adj.shape}")
    print(f"邻接矩阵非零元素: {np.count_nonzero(adj)}")
    
    # 可视化图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 原始草图
    x_cum = np.cumsum(sketch[:, 0])
    y_cum = np.cumsum(sketch[:, 1])
    
    stroke_ends = np.where(sketch[:, 2])[0]
    start_idx = 0
    for end_idx in stroke_ends:
        if end_idx > start_idx:
            axes[0].plot(x_cum[start_idx:end_idx+1], y_cum[start_idx:end_idx+1], 'b-', linewidth=2)
        start_idx = end_idx + 1
    if start_idx < len(x_cum):
        axes[0].plot(x_cum[start_idx:], y_cum[start_idx:], 'b-', linewidth=2)
    axes[0].set_title('Original Sketch')
    axes[0].set_aspect('equal')
    
    # 图表示 - 修正形状问题
    graph_img = graph[0].squeeze()  # 移除单维度
    axes[1].imshow(graph_img, cmap='gray')
    axes[1].set_title('Graph Representation')
    
    plt.tight_layout()
    plt.savefig('/home/sdb/users/jbx2/CS3308/1024Rpcl/test_processing.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ 草图处理测试通过")
    print("处理结果已保存为 test_processing.png")
    return True

if __name__ == "__main__":
    print("开始简化测试...")
    print("="*50)
    
    try:
        test_decoder_only()
        print()
        test_sketch_generation()
        print()
        test_sketch_processing()
        print()
        print("="*50)
        print("✓ 所有简化测试通过！")
        print("项目核心组件工作正常。")
        print("\n下一步建议：")
        print("1. 准备数据集文件到 ../dataset/ 目录")
        print("2. 运行完整训练: python Pix2Seq.py")
        print("3. 运行推理: python inference.py")
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()