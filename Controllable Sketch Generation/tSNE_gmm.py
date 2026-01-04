#!/usr/bin/env python3
"""
t-SNE 可视化 latent 空间 - GMM成分分析版本
在原有类别着色的基础上，增加GMM成分着色分析
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from hyper_params import hp
import glob
import json

# 导入原有的函数
def load_latent_features(data_root="/home/sda/jbx2/CS3308/validate_simple_results/npz/"):
    """
    加载所有类别的隐空间特征
    
    Args:
        data_root: npz 文件根目录
        
    Returns:
        features: (N_samples, 128) 特征矩阵
        labels: (N_samples,) 类别标签
        categories: 类别名称列表
    """
    features = []
    labels = []
    categories = []
    category_to_idx = {}
    
    print("加载隐空间特征...")
    print(f"数据路径: {data_root}")
    
    # 获取所有 npz 文件
    npz_files = glob.glob(os.path.join(data_root, "*.npz"))
    print(f"找到 {len(npz_files)} 个 npz 文件")
    
    if not npz_files:
        print("错误: 没有找到 npz 文件")
        return None, None, None
    
    for i, npz_file in enumerate(npz_files):
        # 提取类别名
        category_name = os.path.basename(npz_file).replace('.npz', '')
        categories.append(category_name)
        category_to_idx[category_name] = i
        
        print(f"处理类别 {i+1}/{len(npz_files)}: {category_name}")
        
        try:
            # 加载 npz 文件
            data = np.load(npz_file, allow_pickle=True)
            
            if 'z' not in data:
                print(f"  警告: {category_name} 没有 'z' 键")
                continue
                
            z_features = data['z']  # 形状: (N_samples, 1, 128) 或 (N_samples, 128)
            print(f"  原始形状: {z_features.shape}")
            
            # 处理不同可能的数据形状
            if len(z_features.shape) == 3:
                # (N_samples, 1, 128) -> (N_samples, 128)
                z_features = z_features.squeeze(1)
            elif len(z_features.shape) == 2 and z_features.shape[1] != 128:
                print(f"  警告: 特征维度不匹配 {z_features.shape[1]} != 128")
                continue
            elif len(z_features.shape) != 2:
                print(f"  警告: 未知的数据形状 {z_features.shape}")
                continue
            
            n_samples = z_features.shape[0]
            print(f"  样本数: {n_samples}, 特征维度: {z_features.shape[1]}")
            
            # 添加到总特征矩阵
            features.append(z_features)
            
            # 创建对应的标签
            labels.extend([i] * n_samples)
            
        except Exception as e:
            print(f"  处理 {category_name} 时出错: {e}")
            continue
    
    if not features:
        print("错误: 没有成功加载任何特征")
        return None, None, None
    
    # 拼接所有特征
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    
    print(f"\n总样本数: {features.shape[0]}")
    print(f"特征维度: {features.shape[1]}")
    print(f"类别数: {len(categories)}")
    print(f"类别列表: {categories}")
    
    return features, labels, categories


def load_gmm_parameters(gmm_param_path="/home/sda/jbx2/CS3308/1024Rpcl/model_save/gmm_parameters/"):
    """
    加载GMM参数
    
    Args:
        gmm_param_path: GMM参数文件路径
        
    Returns:
        gmm_params: 包含GMM参数的字典
        或者 None 如果文件不存在
    """
    print("加载GMM参数...")
    print(f"GMM参数路径: {gmm_param_path}")
    
    # 查找最新的GMM参数文件
    gmm_files = glob.glob(os.path.join(gmm_param_path, "gmm_parameters_epoch_*.npz"))
    
    if not gmm_files:
        print("错误: 没有找到GMM参数文件")
        return None
    
    # 使用最新的文件
    latest_file = max(gmm_files, key=os.path.getctime)
    print(f"使用GMM参数文件: {latest_file}")
    
    try:
        data = np.load(latest_file)
        gmm_params = {
            'de_mu': data['de_mu'],           # (k, z_size) GMM均值
            'de_sigma2': data['de_sigma2'],   # (k, z_size) GMM方差
            'de_alpha': data['de_alpha'],     # (k, 1) GMM权重
            'k': int(data['k']),              # GMM组件数量
            'z_size': int(data['z_size']),    # 隐空间维度
            'epoch': int(data['epoch'])       # 训练epoch
        }
        
        print(f"✓ GMM参数加载成功")
        print(f"  GMM组件数量 (k): {gmm_params['k']}")
        print(f"  隐空间维度 (z_size): {gmm_params['z_size']}")
        print(f"  训练epoch: {gmm_params['epoch']}")
        print(f"  de_mu形状: {gmm_params['de_mu'].shape}")
        print(f"  de_sigma2形状: {gmm_params['de_sigma2'].shape}")
        print(f"  de_alpha形状: {gmm_params['de_alpha'].shape}")
        
        return gmm_params
        
    except Exception as e:
        print(f"加载GMM参数失败: {e}")
        return None


def assign_gmm_components(features, gmm_params):
    """
    为每个特征向量分配GMM组件（基于最大后验概率）
    
    Args:
        features: (N_samples, 128) 特征矩阵
        gmm_params: GMM参数字典
        
    Returns:
        gmm_labels: (N_samples,) 每个样本所属的GMM组件索引
        gmm_probs: (N_samples, k) 每个样本属于各个GMM组件的概率
    """
    print(f"\n分配GMM组件...")
    print(f"特征形状: {features.shape}")
    print(f"GMM组件数: {gmm_params['k']}")
    
    de_mu = gmm_params['de_mu']      # (k, 128)
    de_sigma2 = gmm_params['de_sigma2']  # (k, 128)
    de_alpha = gmm_params['de_alpha']    # (k, 1)
    k = gmm_params['k']
    
    # 计算每个样本属于各个GMM组件的概率
    n_samples = features.shape[0]
    
    # 避免除零和数值不稳定
    eps = 1e-10
    
    # 计算log概率（避免数值下溢）
    log_probs = np.zeros((n_samples, k))
    
    for i in range(k):
        mu_i = de_mu[i]  # (128,)
        sigma2_i = de_sigma2[i]  # (128,)
        alpha_i = de_alpha[i, 0]  # scalar
        
        # 计算每个维度的log概率密度
        diff = features - mu_i  # (n_samples, 128)
        log_exp_part = -0.5 * np.sum((diff**2) / (sigma2_i + eps), axis=1)  # (n_samples,)
        log_frac_part = 0.5 * np.sum(np.log(sigma2_i + eps))  # scalar
        log_norm_part = 0.5 * 128 * np.log(2 * np.pi)  # scalar
        
        # 加上log权重
        log_probs[:, i] = log_exp_part - log_frac_part - log_norm_part + np.log(alpha_i + eps)
    
    # 转换为概率（使用log-sum-exp技巧避免下溢）
    log_probs_max = np.max(log_probs, axis=1, keepdims=True)
    log_probs_stable = log_probs - log_probs_max
    probs = np.exp(log_probs_stable)
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    
    # 分配组件（最大概率）
    gmm_labels = np.argmax(probs, axis=1)
    
    print(f"✓ GMM组件分配完成")
    print(f"  分配结果形状: {gmm_labels.shape}")
    print(f"  概率矩阵形状: {probs.shape}")
    
    # 统计每个组件的样本数
    unique, counts = np.unique(gmm_labels, return_counts=True)
    print(f"  各组件样本分布:")
    for component, count in zip(unique, counts):
        print(f"    组件{component}: {count}个样本 ({count/n_samples*100:.1f}%)")
    
    return gmm_labels, probs


def plot_gmm_components(X_2d, gmm_labels, gmm_params, categories, save_path="./visualization/"):
    """
    绘制GMM成分着色的2D嵌入图
    
    Args:
        X_2d: (N_samples, 2) 2D坐标
        gmm_labels: (N_samples,) GMM组件标签
        gmm_params: GMM参数字典
        categories: 类别名称列表
        save_path: 保存路径
    """
    print(f"\n绘制GMM成分着色图...")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 设置颜色映射 - 使用tab20来支持最多20个组件
    k = gmm_params['k']
    if k <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, k))
    else:
        # 对于k>20的情况，重复使用tab20颜色
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        # 重复使用颜色
        colors = np.tile(colors, (k // 20 + 1, 1))[:k]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 左图：GMM成分着色
    print("  绘制GMM成分着色散点图...")
    for component in range(k):
        # 获取当前组件的样本索引
        mask = gmm_labels == component
        X_component = X_2d[mask]
        
        if len(X_component) > 0:  # 只绘制有样本的组件
            ax1.scatter(X_component[:, 0], X_component[:, 1], 
                       c=[colors[component]], 
                       label=f'GMM {component}',
                       alpha=0.6,
                       s=30,
                       edgecolors='black',  # 添加边缘颜色
                       linewidth=0.5)
    
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax1.set_title(f'GMM Components Visualization (k={k}, epoch={gmm_params["epoch"]})\nSame PCA+t-SNE transformation as category plot', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 右图：类别着色（作为对比）
    print("  绘制类别着色对比图...")
    # 这里需要重新加载类别信息，或者传入类别标签
    # 为了简化，我们创建基于GMM组件的伪类别标签
    # 实际使用时应该传入真实的类别标签
    
    # 简化的对比：显示组件分布
    ax2.hist(gmm_labels, bins=k, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('GMM Component', fontsize=12)
    ax2.set_ylabel('Number of Samples', fontsize=12)
    ax2.set_title('GMM Component Distribution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    timestamp = np.datetime64('now').astype(str).replace(':', '-').replace('T', '_')[:19]
    save_file = os.path.join(save_path, f"gmm_components_tsne_{timestamp}.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    
    # 也保存为PDF
    pdf_file = save_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    
    print(f"✓ GMM成分图已保存")
    print(f"  PNG: {save_file}")
    print(f"  PDF: {pdf_file}")
    
    plt.close()
    
    return save_file


def perform_gmm_tsne_visualization(features, gmm_labels, gmm_params, categories,
                                 pca_components=50,
                                 tsne_perplexity=30,
                                 tsne_learning_rate=200,
                                 tsne_n_iter=2000,
                                 random_state=42,
                                 save_path="./visualization/"):
    """
    执行GMM成分的t-SNE降维和可视化
    
    Args:
        features: (N_samples, 128) 特征矩阵
        gmm_labels: (N_samples,) GMM组件标签
        gmm_params: GMM参数字典
        categories: 类别名称列表
        pca_components: PCA 中间维度
        tsne_perplexity: t-SNE perplexity参数
        tsne_learning_rate: t-SNE 学习率
        tsne_n_iter: t-SNE 迭代次数
        random_state: 随机种子
        save_path: 保存路径
        
    Returns:
        X_2d: (N_samples, 2) 降维后的2D坐标
    """
    print(f"\n{'='*60}")
    print("GMM成分t-SNE可视化")
    print(f"{'='*60}")
    
    print(f"PCA 中间维度: {pca_components}")
    print(f"t-SNE perplexity: {tsne_perplexity}")
    print(f"t-SNE 学习率: {tsne_learning_rate}")
    print(f"t-SNE 迭代次数: {tsne_n_iter}")
    print(f"随机种子: {random_state}")
    print(f"样本数: {features.shape[0]}")
    print(f"GMM组件数: {gmm_params['k']}")
    
    # 步骤1: PCA预处理（与原有可视化使用相同的参数）
    print(f"\n步骤1: PCA 降维 {features.shape[1]} -> {pca_components}")
    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(features)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"  PCA 解释方差比例: {explained_var:.4f} ({explained_var*100:.2f}%)")
    print(f"  PCA 后形状: {X_pca.shape}")
    
    # 步骤2: t-SNE降维
    print(f"\n步骤2: t-SNE 降维 {pca_components} -> 2")
    print(f"  t-SNE 参数: perplexity={tsne_perplexity}, learning_rate={tsne_learning_rate}, n_iter={tsne_n_iter}")
    
    tsne = TSNE(n_components=2, 
                perplexity=tsne_perplexity,
                learning_rate=tsne_learning_rate,
                n_iter=tsne_n_iter,
                random_state=random_state,
                verbose=1)
    
    X_2d = tsne.fit_transform(X_pca)
    
    print(f"  t-SNE 完成！输出形状: {X_2d.shape}")
    
    # 步骤3: 绘制GMM成分可视化
    plot_gmm_components(X_2d, gmm_labels, gmm_params, categories, save_path)
    
    print(f"\n{'='*60}")
    print("GMM成分t-SNE可视化完成！")
    print(f"{'='*60}")
    
    return X_2d


def main():
    """主函数 - 完整的GMM分析流程"""
    print("="*60)
    print("t-SNE GMM成分分析工具")
    print("="*60)
    
    # 参数设置
    latent_data_root = "/home/sda/jbx2/CS3308/batch_z_results/npz/"  # 使用批处理结果
    gmm_param_path = "/home/sda/jbx2/CS3308/1024Rpcl/model_save/gmm_parameters/"
    save_path = "./visualization/"
    
    # t-SNE 参数（与原有可视化保持一致）
    pca_components = 50      # PCA 中间维度
    tsne_perplexity = 30     # t-SNE perplexity
    tsne_learning_rate = 200 # t-SNE 学习率
    tsne_n_iter = 2000       # t-SNE 迭代次数
    random_state = 42        # 随机种子，保证可重复性
    
    print(f"PCA 中间维度: {pca_components}")
    print(f"t-SNE perplexity: {tsne_perplexity}")
    print(f"t-SNE 学习率: {tsne_learning_rate}")
    print(f"t-SNE 迭代次数: {tsne_n_iter}")
    print(f"随机种子: {random_state}")
    
    # 步骤1: 加载隐空间特征
    print(f"\n步骤1: 加载隐空间特征...")
    features, labels, categories = load_latent_features(latent_data_root)
    
    if features is None:
        print("错误: 无法加载特征数据")
        return
    
    # 步骤2: 加载GMM参数
    print(f"\n步骤2: 加载GMM参数...")
    gmm_params = load_gmm_parameters(gmm_param_path)
    
    if gmm_params is None:
        print("错误: 无法加载GMM参数")
        return
    
    # 步骤3: 分配GMM组件
    print(f"\n步骤3: 为每个样本分配GMM组件...")
    gmm_labels, gmm_probs = assign_gmm_components(features, gmm_params)
    
    # 步骤4: 执行t-SNE可视化（使用与原有可视化相同的参数）
    print(f"\n步骤4: 执行t-SNE可视化...")
    X_2d = perform_gmm_tsne_visualization(
        features, gmm_labels, gmm_params, categories,
        pca_components=pca_components,
        tsne_perplexity=tsne_perplexity,
        tsne_learning_rate=tsne_learning_rate,
        tsne_n_iter=tsne_n_iter,
        random_state=random_state,
        save_path=save_path
    )
    
    print(f"\n{'='*60}")
    print("GMM成分t-SNE分析完成！")
    print("="*60)
    
    # 额外分析：比较GMM成分与真实类别的关系
    print(f"\n额外分析：GMM成分与真实类别的关系...")
    print("（注意：这里显示的是GMM成分分布，真实类别需要额外的类别标签）")
    
    # 这里可以添加更多的对比分析
    # 例如：计算每个真实类别中各个GMM成分的分布

if __name__ == '__main__':
    main()
