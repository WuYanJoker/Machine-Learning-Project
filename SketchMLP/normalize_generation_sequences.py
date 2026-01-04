import os
import numpy as np

from Hyper_params import hp
from Utils import off2abs

def normalize_sequence(seq: np.ndarray, seq_len: int) -> np.ndarray:
    """原始为相对坐标：先转绝对坐标，再用 off2abs 归一化到 0~256。
    只改变前两维坐标，第三维状态(0/1/-1)保持不变。
    """
    seq = np.array(seq, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] < 3:
        raise ValueError(f"非法序列形状: {seq.shape}, 期望 (L, >=3)")

    # 只保留前三维
    if seq.shape[1] > 3:
        seq = seq[:, :3]

    # 截断 / 补长
    if seq.shape[0] > seq_len:
        seq = seq[:seq_len]
    elif seq.shape[0] < seq_len:
        pad_len = seq_len - seq.shape[0]
        pad = -np.ones((pad_len, seq.shape[1]), dtype=seq.dtype)
        seq = np.concatenate([seq, pad], axis=0)

    # 以“整行都是 0”作为终止标志；终止之后为 padding（整行 -1）
    zero_mask = np.all(seq == 0, axis=1)
    zero_rows = np.where(zero_mask)[0]
    term_idx = int(zero_rows[0]) if zero_rows.size > 0 else None

    if term_idx is not None:
        # 有显式终止标记：有效长度不包含该行
        valid_len = term_idx
    else:
        # 无终止标记时，仍然允许旧格式：整行 -1 视为 padding 起点
        pad_mask = np.all(seq == -1, axis=1)
        pad_rows = np.where(pad_mask)[0]
        valid_len = int(pad_rows[0]) if pad_rows.size > 0 else seq_len

    # 初始化输出：全部 -1
    out = -np.ones_like(seq, dtype=np.int32)

    if valid_len > 0:
        # 相对坐标 → 绝对坐标（只对前两维）
        seq_abs = seq.copy()
        rel_xy = seq_abs[:valid_len, 0:2]
        abs_xy = np.cumsum(rel_xy, axis=0)
        seq_abs[:valid_len, 0:2] = abs_xy

        # 绝对坐标归一化到 0~256
        seq_norm = off2abs(seq_abs[:valid_len])
        coords = np.clip(seq_norm[:, 0:2], 0.0, 256.0)
        out[:valid_len, 0:2] = np.rint(coords).astype(np.int32)

        # 第三维状态：原样拷贝（0/1/-1），不做任何翻转或裁剪
        out[:valid_len, 2] = seq[:valid_len, 2].astype(np.int32)

    # 终止行：整行 0
    if term_idx is not None and term_idx < seq_len:
        out[term_idx, :] = 0

    # padding 区保持整行 -1
    # 最后一步：除以 1.0，将结果显式转换为浮点类型
    return out.astype(np.float32) / 1.0


def process_npz_file(in_path: str, out_path: str, seq_len: int) -> None:
    data_npz = np.load(in_path, allow_pickle=True)
    if 'sketches' not in data_npz:
        print(f"[跳过] {in_path}: 未找到 'sketches' 键")
        return

    sketches = data_npz['sketches']
    normalized_list = []

    for idx, seq in enumerate(sketches):
        try:
            norm_seq = normalize_sequence(seq, seq_len)
        except Exception as e:
            print(f"[警告] 文件 {os.path.basename(in_path)} 第 {idx} 条序列处理失败: {e}")
            continue
        normalized_list.append(norm_seq)

    if not normalized_list:
        print(f"[警告] {in_path}: 无有效序列，未生成输出")
        return

    normalized_arr = np.stack(normalized_list, axis=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # 保持键名为 'sketches'，便于后续直接使用
    np.savez_compressed(out_path, sketches=normalized_arr)
    print(f"[完成] {in_path} -> {out_path}, 共 {normalized_arr.shape[0]} 条序列")


def main():
    in_root = os.path.join('Data', 'generation', 'coordinate_files')
    out_root = os.path.join('Data', 'generation', 'coordinate_files_normalized')
    seq_len = hp.seq_len

    if not os.path.isdir(in_root):
        raise FileNotFoundError(f"输入目录不存在: {in_root}")

    print(f"输入目录: {in_root}")
    print(f"输出目录: {out_root}")
    print(f"序列长度 (hp.seq_len): {seq_len}")

    for fname in sorted(os.listdir(in_root)):
        if not fname.endswith('.npz'):
            continue
        in_path = os.path.join(in_root, fname)
        base, ext = os.path.splitext(fname)
        out_fname = base + '_norm' + ext
        out_path = os.path.join(out_root, out_fname)

        process_npz_file(in_path, out_path, seq_len)


if __name__ == '__main__':
    main()
