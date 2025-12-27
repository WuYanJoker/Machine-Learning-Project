#!/usr/bin/env python
# map_sketch.py
"""
二维隐空间插值 + 2D 草图阵列可视化
命令行：
    python map_sketch.py airplane angel bus butterfly
"""
import os
import sys
import numpy as np
import cv2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sequence_sketch import (
    load_latent_vectors_by_category,
    interpolate_latent_vectors,
    generate_sketches_from_latent,
    load_model,
)

GRID      = 10
PAD       = 20

def load_four_corners(data_root, c1, c2, c3, c4):
    """返回 4 个 np.ndarray（四角 latent）"""
    corners = []
    for c in [c1, c2, c3, c4]:
        z_list, _ = load_latent_vectors_by_category(data_root, c, c, 1)
        corners.append(z_list[0])          # shape=(D,)
    return corners                         # list[np.ndarray]

def bilinear_grid(z11, z21, z12, z22, grid=GRID):
    """双线性插值生成 grid*grid 个 latent（返回 list[np.ndarray]）"""
    h = np.linspace(0, 1, grid)
    w = np.linspace(0, 1, grid)
    latents = []
    for i, hi in enumerate(h):
        top = interpolate_latent_vectors(z11, z21, grid)   # list 长度=grid
        bot = interpolate_latent_vectors(z12, z22, grid)
        for j, wj in enumerate(w):
            z = interpolate_latent_vectors(top[j], bot[j], grid)[i]
            latents.append(z)
    return latents   # list[np.ndarray] 长度=grid*grid

def draw_canvas(model, latents, grid=GRID, pad=PAD):
    """生成草图并拼成一张大图（numpy uint8）"""
    imgs, _ = generate_sketches_from_latent(model, latents)  # 解包元组
    # 展平嵌套 & 空保护
    if not imgs:
        print("【错误】没有成功生成任何草图，请检查模型输入维度或 latent 格式")
        # 返回一张全白图避免下游崩溃
        return np.ones((grid * (64 + pad) - pad, grid * (64 + pad) - pad), dtype=np.uint8) * 255
    if isinstance(imgs[0], list):
        imgs = [item for sub in imgs for item in sub]
    # 已经是 numpy 数组，无需 convert
    imgs = [np.array(im, dtype=np.uint8) if not isinstance(im, np.ndarray) else im for im in imgs]
    # 打印调试信息
    print(f"[调试] imgs[0] shape: {imgs[0].shape}, len(imgs): {len(imgs)}")
    # 确保是2D图像
    if imgs[0].ndim == 3:
        # 如果是3D (H, W, C)，转为2D灰度图
        imgs = [im.squeeze() if im.shape[-1] == 1 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in imgs]
    H, W = imgs[0].shape
    bigH = grid * (H + pad) - pad
    bigW = grid * (W + pad) - pad
    canvas = np.ones((bigH, bigW), dtype=np.uint8) * 255
    for idx, im in enumerate(imgs):
        i, j = divmod(idx, grid)
        y0 = i * (H + pad)
        x0 = j * (W + pad)
        canvas[y0:y0+H, x0:x0+W] = im
    return canvas

def main():
    if len(sys.argv) != 5:
        print("用法: python map_sketch.py 类别1 类别2 类别3 类别4")
        sys.exit(1)
    c1, c2, c3, c4 = sys.argv[1:5]

    # 自动定位数据集目录
    data_root = "/home/sda/jbx2/CS3308/validate_simple_results_ek34_1225/npz/"
    out_file  = f"map_{c1}_{c2}_{c3}_{c4}.png"

    corners = load_four_corners(data_root, c1, c2, c3, c4)
    z11, z21, z12, z22 = corners
    latents = bilinear_grid(z11, z21, z12, z22, GRID)

    model = load_model(dir_path = "/home/sda/jbx2/CS3308/1024Rpcl/model_save_ek34_1225/")
    canvas = draw_canvas(model, latents)
    cv2.imwrite(os.path.join("1024Rpcl", out_file), canvas)
    print(f"2D 草图阵列已保存 → 1024Rpcl/{out_file}")

if __name__ == "__main__":
    main()