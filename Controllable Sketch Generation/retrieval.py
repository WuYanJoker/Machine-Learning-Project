#!/usr/bin/env python3
"""
修复后的Retrieval评估脚本
处理正确的数据格式 (20, 1, 128)
"""

import torch
import numpy as np
from hyper_params import hp
import os

def evaluate_retrieval_fixed():
    """修复后的检索评估函数"""
    print("="*60)
    print("Retrieval评估 - 修复版本")
    print("="*60)
    
    # 路径设置
    z_root = '/home/sda/jbx2/CS3308/batch_z_results/npz/'
    mask_z_root = '/home/sda/jbx2/CS3308/batch_z_results/retnpz/'
    
    z_code = []
    mask_z_code = []
    category_info = []
    
    print("加载数据...")
    
    # 加载每个类别的数据
    for cat in hp.category:
        print(f'加载 {cat}...')
        
        try:
            # 加载原始特征 - 形状可能是 (N_samples, 128) 或 (N_samples, 1, 128)
            z1 = np.load(z_root + cat, allow_pickle=True)
            z1_data = torch.from_numpy(z1['z'])
            
            # 处理不同可能的数据形状
            if len(z1_data.shape) == 3:
                # (N_samples, 1, 128) -> (N_samples, 128)
                z1_data = z1_data.squeeze(1)
            elif len(z1_data.shape) != 2:
                print(f"  警告: 未知的数据形状 {z1_data.shape}")
                continue
            
            z_code.append(z1_data)
            
            # 加载重建特征 - 同样处理
            z2 = np.load(mask_z_root + cat, allow_pickle=True)
            z2_data = torch.from_numpy(z2['z'])
            
            if len(z2_data.shape) == 3:
                z2_data = z2_data.squeeze(1)
            elif len(z2_data.shape) != 2:
                print(f"  警告: 重建数据形状异常 {z2_data.shape}")
                continue
                
            mask_z_code.append(z2_data)
            
            # 记录类别信息
            category_info.append({
                'name': cat,
                'num_samples': z1_data.shape[0],  # N_samples
                'feature_dim': z1_data.shape[1],  # 128
                'shape': z1_data.shape  # (N_samples, 128)
            })
            
            print(f"  原始特征: {z1_data.shape}")
            print(f"  重建特征: {z2_data.shape}")
            
        except Exception as e:
            print(f"  错误加载 {cat}: {e}")
            continue
    
    if not z_code:
        print("错误: 没有成功加载任何数据")
        return
    
    # 拼接所有类别的特征
    try:
        # 形状: (total_samples, 128) - 已经是2D格式
        z_code_cat = torch.cat(z_code, 0)
        mask_z_code_cat = torch.cat(mask_z_code, 0)
        
        total_samples = z_code_cat.shape[0]
        feature_dim = z_code_cat.shape[1]
        
        print(f"\n总样本数: {total_samples}")
        print(f"特征维度: {feature_dim}")
        print(f"类别数: {len(category_info)}")
        
        # 直接使用，已经是2D张量格式: (total_samples, 128)
        z_code_2d = z_code_cat
        mask_z_code_2d = mask_z_code_cat
        
        print(f"数据形状: {z_code_2d.shape}, {mask_z_code_2d.shape}")
        
        # 移动到GPU
        if torch.cuda.is_available():
            z_code_2d = z_code_2d.cuda()
            mask_z_code_2d = mask_z_code_2d.cuda()
            print("使用CUDA加速")
        else:
            print("使用CPU")
        
    except Exception as e:
        print(f"处理数据时出错: {e}")
        return
    
    # 计算检索准确率
    print("\n计算检索准确率...")
    correct_1 = 0
    correct_10 = 0
    correct_50 = 0
    
    # 逐个样本处理
    for i in range(total_samples):
        if i % 20 == 0:  # 每20个样本打印一次进度
            print(f"  进度: {i}/{total_samples}")
        
        # 获取当前样本特征
        query_feature = z_code_2d[i:i+1]  # (1, 128)
        
        # 计算与所有重建样本的距离
        # query_feature: (1, 128)
        # mask_z_code_2d: (total_samples, 128)
        # 结果: (1, total_samples)
        dist = torch.cdist(query_feature, mask_z_code_2d, p=2).squeeze(0)
        
        # 获取排序索引
        sorted_indices = torch.argsort(dist)
        
        # 评估检索结果
        # Top-1
        if i == sorted_indices[0]:
            correct_1 += 1
        
        # Top-10
        if i in sorted_indices[:10]:
            correct_10 += 1
        
        # Top-50
        if i in sorted_indices[:50]:
            correct_50 += 1
    
    # 计算最终指标
    ret_1 = correct_1 / total_samples
    ret_10 = correct_10 / total_samples
    ret_50 = correct_50 / total_samples
    
    print(f"\n{'='*60}")
    print("Retrieval评估结果:")
    print(f"{'='*60}")
    print(f"总样本数: {total_samples}")
    print(f"类别分布:")
    for info in category_info:
        print(f"  {info['name']}: {info['num_samples']} 样本")
    print(f"\nRet@1:  {ret_1:.4f} ({ret_1*100:.2f}%)")
    print(f"Ret@10: {ret_10:.4f} ({ret_10*100:.2f}%)")
    print(f"Ret@50: {ret_50:.4f} ({ret_50*100:.2f}%)")
    print(f"{'='*60}")
    
    return {
        'total_samples': total_samples,
        'categories': category_info,
        'ret_1': ret_1,
        'ret_10': ret_10,
        'ret_50': ret_50
    }

if __name__ == '__main__':
    evaluate_retrieval_fixed()
