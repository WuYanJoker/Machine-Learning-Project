#!/usr/bin/env python3
"""
隐向量序列插值生成草图
在两个隐向量之间进行插值，生成平滑过渡的草图序列
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from hyper_params import hp
import glob

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from inference import Model
from utils.inference_sketch_processing import draw_three


def load_model(dir_path = "/home/sda/jbx2/CS3308/1024Rpcl/model_save_ek34_1225"):
    """加载预训练模型"""
    print("加载模型...")
    model = Model()
    
    # 查找最新的模型文件
    model_path = dir_path
    encoder_files = [f for f in os.listdir(model_path) if f.startswith('encoderRNN_epoch_') and f.endswith('.pth')]
    
    if not encoder_files:
        print("错误: 未找到模型文件")
        return None
    
    # 使用最新的模型
    latest_epoch = max([int(f.split('_')[2].replace('.pth', '')) for f in encoder_files])
    encoder_path = f"{model_path}/encoderRNN_epoch_{latest_epoch}.pth"
    decoder_path = f"{model_path}/decoderRNN_epoch_{latest_epoch}.pth"
    
    if os.path.exists(encoder_path) and os.path.exists(decoder_path):
        print(f"加载模型: epoch {latest_epoch}")
        model.load(encoder_path, decoder_path)
        return model
    else:
        print(f"错误: 模型文件不存在 {encoder_path} 或 {decoder_path}")
        return None


def load_latent_vectors(data_path, num_samples=1):
    """
    从.npz文件中加载隐向量
    
    Args:
        data_path: npz文件路径
        num_samples: 每个文件要加载的样本数
        
    Returns:
        latent_vectors: 隐向量列表
        categories: 类别名称列表
    """
    print(f"加载隐向量数据: {data_path}")
    
    latent_vectors = []
    categories = []
    
    # 获取所有npz文件
    npz_files = glob.glob(os.path.join(data_path, "*.npz"))
    
    if not npz_files:
        print(f"错误: 在 {data_path} 中未找到npz文件")
        return None, None
    
    print(f"找到 {len(npz_files)} 个npz文件")
    
    for i, npz_file in enumerate(npz_files[:2]):  # 只取前2个文件
        try:
            # 加载npz文件
            data = np.load(npz_file, allow_pickle=True, encoding='latin1')
            
            # 自适应键名
            key = 'z' if 'z' in data else 'train'
            if key not in data:
                print(f"  警告: {npz_file} 没有 'z' 或 'train' 键")
                continue
            
            z_features = data[key]  # 形状可能是 (N_samples, 128) 或 (N_samples, 1, 128)
            
            # 处理不同可能的数据形状
            if len(z_features.shape) == 3:
                z_features = z_features.squeeze(1)  # (N_samples, 1, 128) -> (N_samples, 128)
            
            # 随机选择指定数量的样本
            n_samples_available = z_features.shape[0]
            if n_samples_available < num_samples:
                print(f"  警告: {os.path.basename(npz_file)} 只有 {n_samples_available} 个样本，使用全部")
                selected_indices = list(range(n_samples_available))
            else:
                selected_indices = np.random.choice(n_samples_available, num_samples, replace=False)
            
            # 提取选中的隐向量
            for idx in selected_indices:
                latent_vectors.append(z_features[idx])
                category_name = os.path.basename(npz_file).replace('.npz', '')
                categories.append(category_name)
            
            print(f"  从 {os.path.basename(npz_file)} 加载了 {len(selected_indices)} 个样本")
            
        except Exception as e:
            print(f"  加载 {npz_file} 时出错: {e}")
            continue
    
    if len(latent_vectors) < 2:
        print("错误: 需要至少2个隐向量进行插值")
        return None, None
    
    # 统一转换为 numpy 数组并确保 shape 一致（pad/truncate 到 128 维，匹配模型）
    FIXED_DIM = 128
    padded_vectors = []
    for vec in latent_vectors:
        vec = np.asarray(vec).reshape(-1)
        if len(vec) < FIXED_DIM:
            # pad with zeros
            vec = np.pad(vec, (0, FIXED_DIM - len(vec)), mode='constant')
        elif len(vec) > FIXED_DIM:
            # truncate
            vec = vec[:FIXED_DIM]
        padded_vectors.append(vec)
    latent_vectors = np.stack(padded_vectors)  # (N, FIXED_DIM)
    print(f"总共加载了 {len(latent_vectors)} 个隐向量，shape={latent_vectors.shape}")
    print(f"类别: {categories}")
    
    return latent_vectors, categories


def load_latent_vectors_by_category(data_path, category1, category2, num_samples=1):
    """
    从指定的两个类别中加载隐向量
    
    Args:
        data_path: npz文件根目录
        category1: 第一个类别名称
        category2: 第二个类别名称
        num_samples: 每个类别要加载的样本数
        
    Returns:
        latent_vectors: 隐向量列表
        categories: 类别名称列表
    """
    print(f"\n从指定类别加载隐向量...")
    print(f"类别1: {category1}")
    print(f"类别2: {category2}")
    print(f"每个类别加载样本数: {num_samples}")
    
    latent_vectors = []
    categories = []
    
    # 构建两个类别的文件路径
    category1_file = os.path.join(data_path, f"{category1}.npz")
    category2_file = os.path.join(data_path, f"{category2}.npz")
    
    # 检查文件是否存在
    if not os.path.exists(category1_file):
        print(f"错误: 找不到文件 {category1_file}")
        return None, None
    
    if not os.path.exists(category2_file):
        print(f"错误: 找不到文件 {category2_file}")
        return None, None
    
    # 加载第一个类别的隐向量
    try:
        data1 = np.load(category1_file, allow_pickle=True, encoding='latin1')
        key1 = 'z' if 'z' in data1 else 'train'
        if key1 not in data1:
            print(f"错误: {category1_file} 没有 'z' 或 'train' 键")
            return None, None
        
        z1_features = data1[key1]
        if len(z1_features.shape) == 3:
            z1_features = z1_features.squeeze(1)
        
        # 随机选择指定数量的样本
        n_samples1 = min(num_samples, z1_features.shape[0])
        selected_indices1 = np.random.choice(z1_features.shape[0], n_samples1, replace=False)
        
        for idx in selected_indices1:
            vec = z1_features[idx].squeeze()  # 移除所有大小为1的维度
            if vec.ndim != 1:
                vec = vec.reshape(-1)  # 强制转为1维
            latent_vectors.append(vec)
            categories.append(category1)
        
        print(f"  从 {category1} 加载了 {n_samples1} 个样本")
        
    except Exception as e:
        print(f"  加载 {category1_file} 时出错: {e}")
        return None, None
    
    # 加载第二个类别的隐向量
    try:
        data2 = np.load(category2_file, allow_pickle=True, encoding='latin1')
        key2 = 'z' if 'z' in data2 else 'train'
        if key2 not in data2:
            print(f"错误: {category2_file} 没有 'z' 或 'train' 键")
            return None, None
        
        z2_features = data2[key2]
        if len(z2_features.shape) == 3:
            z2_features = z2_features.squeeze(1)
        
        # 随机选择指定数量的样本
        n_samples2 = min(num_samples, z2_features.shape[0])
        selected_indices2 = np.random.choice(z2_features.shape[0], n_samples2, replace=False)
        
        for idx in selected_indices2:
            vec = z2_features[idx].squeeze()  # 移除所有大小为1的维度
            if vec.ndim != 1:
                vec = vec.reshape(-1)  # 强制转为1维
            latent_vectors.append(vec)
            categories.append(category2)
        
        print(f"  从 {category2} 加载了 {n_samples2} 个样本")
        
    except Exception as e:
        print(f"  加载 {category2_file} 时出错: {e}")
        return None, None
    
    # 统一转换为 numpy 数组并确保 shape 一致（pad/truncate 到 512 维）
    FIXED_DIM = 128
    padded_vectors = []
    for vec in latent_vectors:
        vec = np.asarray(vec).reshape(-1)
        if len(vec) < FIXED_DIM:
            # pad with zeros
            vec = np.pad(vec, (0, FIXED_DIM - len(vec)), mode='constant')
        elif len(vec) > FIXED_DIM:
            # truncate
            vec = vec[:FIXED_DIM]
        padded_vectors.append(vec)
    latent_vectors = np.stack(padded_vectors)  # (N, FIXED_DIM)
    print(f"总共加载了 {len(latent_vectors)} 个隐向量，shape={latent_vectors.shape}")
    
    return latent_vectors, categories


def get_user_input():
    """获取用户输入的类别名称 - 支持命令行参数"""
    import sys
    
    # 获取所有可用的类别
    data_path = "/home/sda/jbx2/CS3308/validate_simple_results_k34_1225/npz/"
    available_categories = []
    npz_files = glob.glob(os.path.join(data_path, "*.npz"))
    for npz_file in npz_files:
        category_name = os.path.basename(npz_file).replace('.npz', '')
        available_categories.append(category_name)
    
    print("可用的类别:", available_categories)
    
    # 检查命令行参数
    if len(sys.argv) >= 3:
        category1 = sys.argv[1]
        category2 = sys.argv[2]
        
        if category1 in available_categories and category2 in available_categories and category1 != category2:
            print(f"从命令行参数获取类别: {category1} -> {category2}")
            return category1, category2
        else:
            print(f"命令行参数错误: '{category1}' 或 '{category2}' 无效或相同")
            print("请提供两个不同的有效类别名称")
            sys.exit(1)
    
    # 交互式输入
    print("请提供两个不同的类别名称")
    while True:
        try:
            category1 = input("请输入第一个类别名称: ").strip()
            if category1 in available_categories:
                break
            else:
                print(f"错误: '{category1}' 不是可用的类别。请从以下中选择: {available_categories}")
        except (EOFError, KeyboardInterrupt):
            print("\n程序被中断")
            sys.exit(1)
    
    while True:
        try:
            category2 = input("请输入第二个类别名称: ").strip()
            if category2 in available_categories and category2 != category1:
                break
            else:
                if category2 == category1:
                    print("错误: 两个类别不能相同。请选择不同的类别。")
                else:
                    print(f"错误: '{category2}' 不是可用的类别。请从以下中选择: {available_categories}")
        except (EOFError, KeyboardInterrupt):
            print("\n程序被中断")
            sys.exit(1)
    
    return category1, category2


def interpolate_latent_vectors(z1, z2, num_interpolations=10):
    """
    在两个隐向量之间进行线性插值
    
    Args:
        z1: 第一个隐向量 (128,)
        z2: 第二个隐向量 (128,)
        num_interpolations: 插值点数量（包括起点和终点）
        
    Returns:
        interpolated_z: 插值后的隐向量序列 (num_interpolations, 128)
    """
    print(f"\n在隐向量之间进行插值...")
    print(f"插值点数量: {num_interpolations}")
    
    # 生成插值权重
    alphas = np.linspace(0, 1, num_interpolations)
    
    # 线性插值
    interpolated_z = []
    for alpha in alphas:
        z_interp = (1 - alpha) * z1 + alpha * z2
        interpolated_z.append(z_interp)
    
    interpolated_z = np.array(interpolated_z)
    
    print(f"插值完成，形状: {interpolated_z.shape}")
    
    return interpolated_z


def generate_sketches_from_latent(model, latent_vectors, save_dir="./sequence_results"):
    """
    从隐向量生成草图序列
    
    Args:
        model: 预训练模型
        latent_vectors: 隐向量序列 (N, 128)
        save_dir: 保存目录
        
    Returns:
        generated_images: 生成的草图图像列表
        sequence_info: 序列信息
    """
    print(f"\n从隐向量生成草图...")
    print(f"输入隐向量数量: {len(latent_vectors)}")
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    generated_images = []
    sequence_info = []
    
    # 设置模型为评估模式
    model.encoder.eval()
    model.decoder.eval()
    
    with torch.no_grad():
        for i, z in enumerate(latent_vectors):
            print(f"生成第 {i+1}/{len(latent_vectors)} 个草图...")
            
            try:
                # 将numpy数组转换为torch张量
                z_tensor = torch.from_numpy(z).float()
                if hp.use_cuda:
                    z_tensor = z_tensor.cuda()
                
                # 添加batch维度
                z_tensor = z_tensor.unsqueeze(0)  # (1, 128)
                
                # 使用conditional_generate_by_z生成草图序列
                generated_sequence = model.conditional_generate_by_z(z_tensor, index=i)
                
                if len(generated_sequence) == 0:
                    print(f"  警告: 第{i+1}个生成的序列为空")
                    continue
                
                # 渲染成图片
                sketch_image = draw_three(generated_sequence, img_size=256, show=False)
                
                if sketch_image is None or sketch_image.size == 0:
                    print(f"  警告: 第{i+1}个渲染失败")
                    continue
                
                # 保存图片
                filename = f"sketch_{i:03d}.png"
                save_path = os.path.join(save_dir, filename)
                cv2.imwrite(save_path, sketch_image)
                
                generated_images.append(sketch_image)
                sequence_info.append({
                    'index': i,
                    'filename': filename,
                    'sequence_length': len(generated_sequence),
                    'sequence_shape': generated_sequence.shape
                })
                
                print(f"  ✓ 生成并保存: {filename}")
                
            except Exception as e:
                print(f"  生成第{i+1}个时出错: {e}")
                continue
    
    print(f"\n生成完成！共生成 {len(generated_images)} 个草图")
    
    return generated_images, sequence_info


def create_sequence_collage(generated_images, categories, save_dir="./sequence_results"):
    """
    创建插值序列的拼接图
    
    Args:
        generated_images: 生成的草图图像列表
        categories: 原始类别信息
        save_dir: 保存目录
        
    Returns:
        collage_path: 拼接图路径
    """
    print(f"\n创建序列拼接图...")
    
    if len(generated_images) == 0:
        print("没有图片可以拼接")
        return None
    
    # 计算网格布局
    n_images = len(generated_images)
    n_cols = min(5, n_images)  # 每行最多5张图片
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # 获取单张图片的尺寸
    h, w = generated_images[0].shape[:2]
    
    # 创建大图
    collage = np.ones((n_rows * h, n_cols * w, 3), dtype=np.uint8) * 255
    
    # 填充图片
    for i, img in enumerate(generated_images):
        row = i // n_cols
        col = i % n_cols
        collage[row*h:(row+1)*h, col*w:(col+1)*w] = img
    
    # 保存拼接图
    timestamp = np.datetime64('now').astype(str).replace(':', '-').replace('T', '_')[:19]
    collage_filename = f"sequence_collage_{timestamp}.png"
    collage_path = os.path.join(save_dir, collage_filename)
    cv2.imwrite(collage_path, collage)
    
    print(f"✓ 拼接图已保存: {collage_path}")
    print(f"  尺寸: {collage.shape[1]}x{collage.shape[0]} 像素")
    print(f"  包含 {n_images} 张图片 ({n_rows}行 x {n_cols}列)")
    
    return collage_path


def main(category1, category2, data_path = "/home/sda/jbx2/CS3308/validate_simple_results_ek34_1225/npz/", dir_path="/home/sda/jbx2/CS3308/1024Rpcl/model_save_ek34_1225/", num_interpolations = 10):
    """主函数"""
    print("="*60)
    print("隐向量序列插值生成草图")
    print("="*60)
    
    # 参数设置
    data_path = data_path
    save_dir = "./sequence_results_k34"
    num_samples_per_category = 1  # 每个类别提取1个样本
    num_interpolations = num_interpolations   # 插值数量（包括起点和终点）
    
    print(f"数据路径: {data_path}")
    print(f"保存目录: {save_dir}")
    print(f"每个类别提取样本数: {num_samples_per_category}")
    print(f"插值数量: {num_interpolations}")
    
    # 步骤1: 加载模型
    print(f"\n步骤1: 加载模型...")
    model = load_model(dir_path=dir_path)
    if model is None:
        print("无法加载模型，程序终止")
        return
    
    # 步骤2: 从指定类别加载隐向量数据
    print(f"\n步骤2: 从指定类别加载隐向量数据...")
    latent_vectors, categories = load_latent_vectors_by_category(data_path, category1, category2, num_samples_per_category)
    
    if latent_vectors is None or len(latent_vectors) < 2:
        print("需要至少2个隐向量进行插值，程序终止")
        return
    
    # 步骤3: 选择两个不同的隐向量进行插值
    print(f"\n步骤3: 选择隐向量进行插值...")
    print(f"可用隐向量: {len(latent_vectors)} 个")
    print(f"类别: {categories}")
    
    # 选择第一个和最后一个（确保来自不同类别）
    z1 = latent_vectors[0]
    z2 = latent_vectors[-1]
    
    print(f"选择的隐向量:")
    print(f"  起点: {categories[0]} (类别0)")
    print(f"  终点: {categories[-1]} (类别{len(categories)-1})")
    print(f"  起点向量形状: {z1.shape}")
    print(f"  终点向量形状: {z2.shape}")
    
    # 步骤4: 隐向量插值
    print(f"\n步骤4: 隐向量插值...")
    interpolated_z = interpolate_latent_vectors(z1, z2, num_interpolations)
    
    # 步骤5: 生成草图序列
    print(f"\n步骤5: 生成草图序列...")
    generated_images, sequence_info = generate_sketches_from_latent(model, interpolated_z, save_dir)
    
    # 步骤6: 创建拼接图
    if len(generated_images) > 0:
        print(f"\n步骤6: 创建拼接图...")
        collage_path = create_sequence_collage(generated_images, categories, save_dir)
        
        # 保存序列信息
        info_file = os.path.join(save_dir, "sequence_info.txt")
        with open(info_file, 'w') as f:
            f.write("隐向量序列插值生成信息\n")
            f.write("="*50 + "\n")
            f.write(f"起点类别: {categories[0]}\n")
            f.write(f"终点类别: {categories[-1]}\n")
            f.write(f"插值数量: {num_interpolations}\n")
            f.write(f"生成图片数量: {len(generated_images)}\n")
            f.write(f"保存目录: {save_dir}\n")
            f.write(f"生成时间: {np.datetime64('now').astype(str)}\n")
            f.write("\n序列信息:\n")
            for info in sequence_info:
                f.write(f"  图片{info['index']}: {info['filename']} "
                       f"(序列长度: {info['sequence_length']})\n")
        
        print(f"\n序列信息已保存到: {info_file}")
    
    print(f"\n{'='*60}")
    print("序列生成完成！")
    print(f"所有结果保存在: {save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    # 在主函数中直接设置两个类别变量
    # 你可以在这里修改为你想要的任何两个不同类别
    category1 = "airplane"  # 第一个类别
    category2 = "angel"  # 第二个类别
    
    # 验证类别是否有效
    data_path = "/home/sda/jbx2/CS3308/validate_simple_results_ek34_1225/npz/"
    available_categories = []
    npz_files = glob.glob(os.path.join(data_path, "*.npz"))
    for npz_file in npz_files:
        category_name = os.path.basename(npz_file).replace('.npz', '')
        available_categories.append(category_name)
    
    if category1 not in available_categories:
        print(f"错误: '{category1}' 不是可用的类别。可用类别: {available_categories}")
        sys.exit(1)
    
    if category2 not in available_categories:
        print(f"错误: '{category2}' 不是可用的类别。可用类别: {available_categories}")
        sys.exit(1)
    
    if category1 == category2:
        print("错误: 两个类别不能相同。请选择不同的类别。")
        sys.exit(1)
    
    print(f"选择的类别: {category1} -> {category2}")
    
    # 执行主函数
    main(category1, category2, num_interpolations=10)
