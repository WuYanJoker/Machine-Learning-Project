#!/usr/bin/env python3
"""
t-SNE 可视化 latent 空间
将 128 维特征降维到 2D 并可视化
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from hyper_params import hp
import glob


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


def perform_tsne_visualization(features, labels, categories, 
                             pca_components=50,
                             tsne_perplexity=30,
                             tsne_learning_rate=200,
                             tsne_n_iter=2000,
                             random_state=42):
    """
    执行 t-SNE 降维和可视化
    
    Args:
        features: (N_samples, 128) 特征矩阵
        labels: (N_samples,) 类别标签
        categories: 类别名称列表
        pca_components: PCA 降维到的中间维度
        tsne_perplexity: t-SNE perplexity 参数
        tsne_learning_rate: t-SNE 学习率
        tsne_n_iter: t-SNE 迭代次数
        random_state: 随机种子，保证结果可重复
        
    Returns:
        X_2d: (N_samples, 2) 降维后的 2D 坐标
    """
    print(f"\n开始降维...")
    print(f"步骤1: PCA 降维 {features.shape[1]} -> {pca_components}")
    
    # Step 1: PCA 预处理
    pca = PCA(n_components=pca_components, random_state=random_state)
    X_pca = pca.fit_transform(features)
    
    explained_var = np.sum(pca.explained_variance_ratio_)
    print(f"  PCA 解释方差比例: {explained_var:.4f} ({explained_var*100:.2f}%)")
    print(f"  PCA 后形状: {X_pca.shape}")
    
    print(f"步骤2: t-SNE 降维 {pca_components} -> 2")
    print(f"  t-SNE 参数: perplexity={tsne_perplexity}, learning_rate={tsne_learning_rate}, n_iter={tsne_n_iter}")
    
    # Step 2: t-SNE 降维
    tsne = TSNE(n_components=2, 
                perplexity=tsne_perplexity,
                learning_rate=tsne_learning_rate,
                n_iter=tsne_n_iter,
                random_state=random_state,
                verbose=1)
    
    X_2d = tsne.fit_transform(X_pca)
    
    print(f"  t-SNE 完成！输出形状: {X_2d.shape}")
    
    return X_2d


def plot_2d_embedding(X_2d, labels, categories, save_path="./visualization/"):
    """
    绘制 2D 嵌入图
    
    Args:
        X_2d: (N_samples, 2) 2D 坐标
        labels: (N_samples,) 类别标签
        categories: 类别名称列表
        save_path: 保存路径
    """
    print(f"\n绘制 2D 嵌入图...")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 设置颜色映射
    n_categories = len(categories)
    colors = plt.cm.tab10(np.linspace(0, 1, n_categories))
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 为每个类别绘制散点
    for i, category in enumerate(categories):
        # 获取当前类别的样本索引
        mask = labels == i
        X_cat = X_2d[mask]
        
        plt.scatter(X_cat[:, 0], X_cat[:, 1], 
                   c=[colors[i]], 
                   label=category,
                   alpha=0.6,
                   s=30)
        
        print(f"  类别 {category}: {X_cat.shape[0]} 个样本")
    
    # 图形美化
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('Latent Space Visualization with t-SNE\n(128D → 50D PCA → 2D t-SNE)', fontsize=14)
    
    # 图例设置
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    # 保存图像
    timestamp = np.datetime64('now').astype(str).replace(':', '-').replace('T', '_')[:19]
    save_file = os.path.join(save_path, f"tsne_visualization_{timestamp}.png")
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"✓ 图像已保存: {save_file}")
    
    # 也保存为PDF（矢量格式）
    pdf_file = save_file.replace('.png', '.pdf')
    plt.savefig(pdf_file, bbox_inches='tight')
    print(f"✓ PDF 已保存: {pdf_file}")
    
    # 显示图形（如果在交互环境）
    try:
        plt.show()
    except:
        print("图形已保存，无法在非交互环境显示")
    
    plt.close()
    
    return save_file


def main():
    """主函数"""
    print("="*60)
    print("t-SNE Latent Space 可视化")
    print("="*60)
    
    # 参数设置
    data_root = "/home/sda/jbx2/CS3308/batch_z_results/npz"
    save_path = "./visualization/"
    
    # t-SNE 参数
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
    
    # 步骤1: 加载特征数据
    features, labels, categories = load_latent_features(data_root)
    
    if features is None:
        print("错误: 无法加载特征数据")
        return
    
    # 步骤2: 执行 t-SNE 降维
    X_2d = perform_tsne_visualization(
        features, labels, categories,
        pca_components=pca_components,
        tsne_perplexity=tsne_perplexity,
        tsne_learning_rate=tsne_learning_rate,
        tsne_n_iter=tsne_n_iter,
        random_state=random_state
    )
    
    # 步骤3: 绘制和保存可视化
    plot_2d_embedding(X_2d, labels, categories, save_path)
    
    print(f"\n{'='*60}")
    print("t-SNE 可视化完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
