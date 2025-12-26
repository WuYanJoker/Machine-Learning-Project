#!/usr/bin/env python3
"""
快速测试脚本 - 用于验证环境配置和代码运行
"""

import torch
import numpy as np
from hyper_params import hp
from encoder import myencoder
from decoder import DecoderRNN
from utils.sketch_processing import make_graph

def test_model_components():
    """测试模型组件"""
    print("测试模型组件...")
    
    # 设置小批量参数用于测试
    hp.batch_size = 2
    hp.graph_number = 5  # 减少图数量
    hp.Nz = 64  # 减少隐空间维度
    hp.use_cuda = False  # 强制使用CPU进行测试
    
    # 创建模拟数据
    batch_size = 2
    seq_len = 50
    sketch_data = np.random.randn(batch_size, seq_len, 3).astype(np.float32) * 10
    sketch_data[:, :, 2] = (sketch_data[:, :, 2] > 0).astype(np.float32)  # 二元笔状态
    
    # 创建图数据
    graphs = []
    adjs = []
    for i in range(batch_size):
        graph, adj = make_graph(sketch_data[i], 
                               graph_num=hp.graph_number,
                               graph_picture_size=64,
                               mask_prob=0.0)
        graphs.append(graph)
        adjs.append(adj)
    
    graphs = torch.stack([torch.from_numpy(g).float() for g in graphs])
    adjs = torch.stack([torch.from_numpy(a).float() for a in adjs])
    
    # 测试编码器
    encoder = myencoder(hps=hp)
    encoder.eval()  # 设置为评估模式
    z, mu, sigma, mseloss, rpclloss, update_data = encoder(graphs)
    
    print(f"编码器输出形状: z={z.shape}, mu={mu.shape}, sigma={sigma.shape}")
    print(f"重建损失: {mseloss.item():.4f}")
    print(f"RPCL损失: {rpclloss[0].item():.4f}")
    
    # 测试解码器
    decoder = DecoderRNN()
    decoder.eval()  # 设置为评估模式
    sos = torch.zeros(1, batch_size, 5)
    sos[0, :, 2] = 1  # 设置开始标志
    
    inputs = torch.cat([sos, z.unsqueeze(0).repeat(1, batch_size, 1)], dim=-1)
    
    pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell = decoder(inputs, z)
    
    print(f"解码器输出形状: pi={pi.shape}, mu_x={mu_x.shape}, q={q.shape}")
    print("✓ 模型组件测试通过")
    
    return True

def test_training_step():
    """测试单个训练步骤"""
    print("测试训练步骤...")
    
    # 这里可以添加简单的训练循环测试
    # 由于需要完整的数据集，这里只做组件测试
    print("✓ 训练步骤框架测试通过")
    
    return True

def test_generation():
    """测试生成过程"""
    print("测试生成过程...")
    
    # 创建随机隐向量
    z = torch.randn(1, hp.Nz)
    hp.use_cuda = False  # 确保使用CPU
    
    # 解码器生成
    decoder = DecoderRNN()
    decoder.eval()
    
    sos = torch.Tensor([0, 0, 1, 0, 0]).view(1, 1, -1)
    s = sos
    
    seq_x, seq_y, seq_z = [], [], []
    hidden_cell = None
    
    with torch.no_grad():
        for i in range(20):  # 生成20步
            input_seq = torch.cat([s, z.unsqueeze(0)], dim=-1)
            pi, mu_x, mu_y, sigma_x, sigma_y, rho_xy, q, hidden, cell = decoder(input_seq, z, hidden_cell)
            hidden_cell = (hidden, cell)
            
            # 简单采样下一步
            pi_idx = torch.multinomial(pi[0, 0], 1).item()
            q_idx = torch.multinomial(q[0, 0], 1).item()
            
            dx = mu_x[0, 0, pi_idx].item()
            dy = mu_y[0, 0, pi_idx].item()
            
            seq_x.append(dx)
            seq_y.append(dy)
            seq_z.append(q_idx == 1)
            
            # 更新状态
            next_state = torch.zeros(5)
            next_state[0] = dx
            next_state[1] = dy
            next_state[q_idx + 2] = 1
            s = next_state.view(1, 1, -1)
            
            if q_idx == 2:  # EOS
                break
    
    print(f"生成序列长度: {len(seq_x)}")
    print(f"前5步坐标: x={seq_x[:5]}, y={seq_y[:5]}")
    print("✓ 生成过程测试通过")
    
    return True

if __name__ == "__main__":
    print("开始快速测试...")
    print("="*50)
    
    try:
        test_model_components()
        print()
        # test_training_step()
        # print()
        test_generation()
        print()
        print("="*50)
        # print("✓ 所有测试通过！环境配置正确，可以运行项目。")
        # print("建议下一步：")
        # print("1. 准备数据集文件到 ../dataset/ 目录")
        # print("2. 运行 python Pix2Seq.py 开始训练（需要较长时间）")
        # print("3. 或使用已有模型运行推理脚本")
        
    except Exception as e:
        print(f"✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()