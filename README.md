# Machine-Learning-Project 总览

本仓库包含两个相对独立但互补的草图相关子项目：

- **Controllable Sketch Generation**：基于变分自编码器、混合高斯与序列解码的草图生成 / 补全 / 检索与可视化项目（课程设计在原项目基础上的增强版）。
- **SketchMLP**：论文 *"SketchMLP: Effectively Utilize Rasterized Images and Drawing Sequences for Sketch Recognition"* 的实现，用于草图分类与识别。

两个子项目均基于 QuickDraw-414k 系列数据集，一个侧重生成与理解，另一个侧重识别与分类，可单独使用，也可以联合用于端到端的“生成–识别”实验。

---

## 目录结构

```text
Machine-Learning-Project/
├── Controllable Sketch Generation/   # 草图生成与检索（1024Rpcl 扩展版）
│   ├── Pix2Seq.py                    # 主训练脚本
│   ├── inference.py                  # 生成与推理
│   ├── complete_sketch.py           # 草图补全
│   ├── sequence_sketch.py           # 序列式生成
│   ├── map_sketch.py                # 草图阵列与可视化
│   ├── retrieval.py                 # 草图检索与评估
│   ├── tSNE.py / tSNE_gmm.py        # t-SNE 与 GMM 可视化
│   ├── view_results.py              # 结果查看
│   ├── hyper_params.py              # 超参数配置
│   ├── model_save_ek20_1227/        # 已训练模型与 GMM 参数
│   ├── utils/                       # 草图预处理与推理工具
│   └── requirements.txt
├── SketchMLP/                        # 草图识别模型（SketchMLP）
│   ├── Train.py / Eval.py           # 训练与评估脚本
│   ├── Networks5.py                 # 模型结构
│   ├── Dataset.py / SketchUtils.py  # 数据集与工具
│   ├── Hyper_params.py              # 训练配置
│   ├── test_generation.py           # 小规模测试 / 验证脚本
│   ├── visualize_generation.py      # 结果可视化脚本
│   ├── normalize_generation_sequences.py  # 序列归一化工具
│   ├── data_processing.ipynb        # QuickDraw 预处理
│   ├── requirements.txt
│   └── README.md
└── README.md                        # 当前总览文档
```

详细说明、更多脚本和可视化代码，请分别查看各子目录中的 README 和源码。

---

## 环境与依赖

推荐环境：

- Python ≥ 3.8
- PyTorch（建议使用支持 CUDA 的 GPU 版本）
- CUDA（可选，但强烈推荐用于训练生成模型和大规模识别）

每个子项目都有独立的依赖文件，建议为每个子项目创建单独的虚拟环境：

```bash
# 进入草图生成项目
cd "Controllable Sketch Generation"
pip install -r requirements.txt

# 进入草图识别项目
cd ../SketchMLP
pip install -r requirements.txt
```

> 建议使用 `conda` 或 `venv` 创建虚拟环境，以避免包版本冲突。

---

## 子项目一：Controllable Sketch Generation

位于目录：`Controllable Sketch Generation/`

该项目基于 RPCL 与 Pix2Seq 思路，通过 VAE + GMM + 序列解码器，对手绘草图进行：

- 无监督表征学习与隐空间建模
- 条件/无条件草图生成
- 草图补全与序列式生成
- 相似草图检索
- t-SNE / GMM 等可视化分析

相比原始实现，本仓库版本在课程设计复现过程中新增/增强了：

- `complete_sketch.py`：对不完整草图进行补全
- `sequence_sketch.py`：按时间步逐步生成草图序列
- `map_sketch.py` / `view_results.py`：生成结果的排布与可视化
- `tSNE.py` / `tSNE_gmm.py`：隐空间与 GMM 结果的可视化
- 在 `inference.py` 与 `retrieval.py` 中增加了大量中间结果输出与可视化

### 数据准备（概要）

该项目使用 QuickDraw 的 `.npz` 草图数据，一般放在：

- `Controllable Sketch Generation/../dataset/` 目录下

数据格式通常为形状为 `(N, 3)` 的数组，其中两列为坐标增量，一列为笔状态（移动/落笔/抬笔）。
具体类别与文件命名方式，请参考 `Controllable Sketch Generation/README.md` 和数据处理脚本。

### 快速上手

进入子目录后，可按以下流程使用（示例）：

```bash
cd "Controllable Sketch Generation"

# 1. 训练模型
python Pix2Seq.py

# 2. 使用已训练模型进行草图生成
python inference.py

# 3. 草图检索评估
python retrieval.py

# 4. 其它实验（补全 / t-SNE / 结果可视化等）
python complete_sketch.py
python tSNE.py
python view_results.py
```

更多细节（核心原理、关键参数 `hyper_params.py`、可选可视化等），请阅读
`Controllable Sketch Generation/README.md`。

---

## 子项目二：SketchMLP

位于目录：`SketchMLP/`

该项目是论文 *SketchMLP* 的实现，用于高效利用：

- **光栅化图像**（渲染后的草图图像）
- **绘制序列**（笔画顺序与轨迹）

来进行草图识别 / 分类实验。

在原始实现的基础上，本仓库对 SketchMLP 做了以下实用扩展：

- `test_generation.py`：加载训练好的模型，在一小部分数据上快速测试/验证，便于检查模型是否训练成功、代码是否正常运行。
- `visualize_generation.py`：对模型预测结果进行可视化（如类别预测、混淆矩阵或示例草图），并将图像输出到 `visualization/` 等目录，方便做定性分析。
- `normalize_generation_sequences.py`：提供对绘制序列的归一化与预处理函数，可在训练、评估或可视化前统一处理输入序列。

### 数据与预处理（概要）

- 从 [QuickDraw 官方](https://quickdraw.withgoogle.com/data) 下载原始 `.npy` 或 `.npz` 数据，放入 `./QuickDraw`。
- 使用 `data_processing.ipynb` 进行预处理，生成训练/验证/测试数据。
- 可选地下载 QuickDraw414k 扩展数据集，解压到 `SketchMLP/Dataset/`。

项目作者在子目录 README 中提供了百度网盘链接，包含：

- 预处理好的 QuickDraw 与 QuickDraw414k 数据
- 预训练权重与对应的 `Hyper_params.py` 配置

### 训练与测试（示例）

```bash
cd SketchMLP

# 1. 根据需要修改训练配置
vim Hyper_params.py   # 或在编辑器中直接修改

# 2. 训练
python -u Train.py

# 3. 评估
# 将模型放入 ./pretrain 并重命名为 QD.pkl 或 QD414k.pkl
python -u Eval.py

# 4. 使用本仓库新增的辅助脚本（可选）

# 小规模快速测试（例如检查模型/环境是否正常）
python -u test_generation.py

# 生成并保存可视化结果（如预测示例、曲线等）
python -u visualize_generation.py
```

更多安装说明、数据下载链接与可视化结果，请查看
`SketchMLP/README.md`。

---

## 数据与实验建议

- **环境隔离**：
	- 生成项目（Controllable Sketch Generation）依赖的 PyTorch 与可视化工具版本可能与 SketchMLP 略有不同。
	- 建议为两个子项目分别创建虚拟环境，确保依赖清晰可控。
- **GPU 使用**：
	- 若显存较小，可适当减小 batch size、图像/序列长度等参数。
	- 训练前请确认 CUDA 与对应的 PyTorch 版本匹配。

---

## 参考与致谢

- Controllable Sketch Generation 子项目基于 RPCL 与 Pix2Seq 相关工作，并在原实现基础上进行了测试与可视化扩展。
- SketchMLP 子项目实现了论文 *"SketchMLP: Effectively Utilize Rasterized Images and Drawing Sequences for Sketch Recognition"* 中提出的模型结构与训练流程。

如在复现或实验过程中遇到问题，建议：

- 先在各子项目目录下单独运行最小示例（如简化训练 / 推理脚本）。
- 根据报错信息检查 Python 版本、依赖包版本以及 CUDA 配置。
