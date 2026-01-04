# 1024Rpcl - 基于RPCL的草图生成项目

## 项目简介

1024Rpcl是一个基于Pix2Seq框架的草图生成和理解项目，使用Rival Penalized Competitive Learning (RPCL)算法进行无监督学习，能够从隐空间生成高质量的手绘草图。

## 核心原理

### 技术架构
1. **RPCL-VAE编码器**：使用34个高斯混合成分的变分自编码器，采用RPCL算法进行聚类学习
2. **图神经网络**：将草图转换为21个节点的图结构（1个全局图+20个局部图）
3. **LSTM解码器**：基于序列到序列的混合高斯分布生成模型

### 关键特性
- **无监督学习**：不需要标注数据，自动学习草图的隐空间表示
- **高质量生成**：生成的草图保持原始风格和结构
- **多样性控制**：通过调节温度参数控制生成多样性
- **检索能力**：支持基于隐空间向量的相似草图检索

## 环境配置

### 依赖安装
```bash
pip install -r requirements.txt
```

### 目录结构
```
1024Rpcl/
├── Pix2Seq.py                    # 主训练脚本
├── complete_sketch.py            # 草图补全脚本
├── decoder.py                    # LSTM解码器
├── encoder.py                    # RPCL-VAE编码器
├── hyper_params.py               # 超参数配置
├── inference.py                  # 推理和生成脚本
├── map_sketch.py                 # 草图阵列生成
├── retrieval.py                  # 草图检索评估
├── sequence_sketch.py            # 草图序列生成
├── tSNE.py                       # t-SNE可视化
├── tSNE_gmm.py                   # GMM t-SNE可视化
├── test.py                       # 测试脚本
├── utils/
│   ├── __init__.py                       # 工具包初始化
│   ├── (seed.npy)                        # 种子，由于文件大小限制没有上传
│   ├── inference_sketch_processing.py    # 推理处理工具
│   └── sketch_processing.py              # 草图处理工具
├── view_results.py               # 结果查看工具
├── requirements.txt              # 依赖包
└── README.md                     # 项目说明
```

在课程设计复现该项目过程中，相比于原项目增加了测试和可视化代码如下：
1. complete_sketch.py
2. map_sketch.py
3. sequence_sketch.py
4. tSNE.py
5. tSNE_gmm.py
6. view_results.py
7. inference.py中添加诸多输出和可视化功能
8. retrieval.py中与前面已修改代码做适配

## 快速开始

### 1. 环境测试
运行简化测试验证环境配置：
```bash
python simple_test.py
```

### 2. 数据准备
项目使用QuickDraw数据集的.npz格式文件，需要准备以下类别的数据文件到`../dataset/`目录：
- airplane.npz, angel.npz, alarm clock.npz, apple.npz
- butterfly.npz, belt.npz, bus.npz
- cake.npz, cat.npz, clock.npz, eye.npz, fish.npz
- pig.npz, sheep.npz, spider.npz, The Great Wall of China.npz
- umbrella.npz

### 3. 模型训练
```bash
python Pix2Seq.py
```
训练过程会自动保存模型到`model_save/`目录。

### 4. 草图生成
使用训练好的模型进行草图生成：
```bash
python inference.py
```

### 5. 检索评估
运行检索性能评估：
```bash
python retrieval.py
```

## 核心参数配置

在`hyper_params.py`中可以调整的关键参数：

```python
# 模型架构
self.Nz = 128              # 隐空间维度
self.M = 20                # 混合高斯成分数
self.graph_number = 21     # 图节点数（1+20）
self.graph_picture_size = 128  # 图像块大小

# 训练参数
self.batch_size = 200      # 批次大小
self.lr = 0.001            # 学习率
self.max_seq_length = 200  # 最大序列长度

# RPCL参数
self.br = 30000            # RPCL开始迭代数
```

## 技术细节

### 数据格式
输入草图格式为numpy数组，形状为(N, 3)，其中：
- 第0列：Δx (x坐标增量)
- 第1列：Δy (y坐标增量)  
- 第2列：笔状态 (0:移动, 1:落笔, 2:抬笔)

### 模型输出
解码器输出包含：
- **混合权重** (π)：20个高斯成分的权重
- **位置参数** (μx, μy)：每个成分的均值
- **尺度参数** (σx, σy)：每个成分的标准差
- **相关系数** (ρxy)：x和y的相关性
- **笔状态** (q)：三种笔状态的概率

## 性能优化建议

1. **GPU加速**：确保CUDA可用，自动检测GPU
2. **批次大小**：根据GPU内存调整batch_size
3. **学习率调度**：使用自适应学习率衰减
4. **梯度裁剪**：防止梯度爆炸，默认clip值为1.0

## 故障排除

### 常见问题

1. **设备不匹配错误**：
   - 确保`hp.use_cuda`设置正确
   - 检查CUDA是否可用

2. **内存不足**：
   - 减小`batch_size`
   - 减小`graph_picture_size`
   - 使用更少的`graph_number`

3. **生成质量差**：
   - 调整`temperature`参数
   - 增加训练迭代次数
   - 检查数据预处理

### 测试验证

如果运行测试失败，请检查：
1. PyTorch版本兼容性
2. 依赖包是否正确安装
3. CUDA环境（如使用GPU）

## 扩展应用

该项目可以扩展到：
- **风格迁移**：在不同风格间转换草图
- **草图补全**：根据部分输入完成整个草图
- **条件生成**：基于文本描述生成草图
- **交互式编辑**：实时草图编辑和生成

## 引用

如果您使用该项目，请引用相关论文：
```bibtex
@inproceedings{rpcl-sketch-2024,
  title={RPCL-based Sketch Generation and Understanding},
  author={Your Name},
  booktitle={Conference},
  year={2024}
}
```

## 联系方式

如有问题或建议，请提交Issue或Pull Request。