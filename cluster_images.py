#!/usr/bin/env python3
"""使用 ResNet50 提取图片特征，并通过 K-Means 自动分组。"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import KMeans
from tqdm import tqdm
from torchvision import models, transforms

try:
    # 注册 HEIC/HEIF 解码器；未安装 pillow-heif 时仍可处理其他图片格式。
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

# 递归扫描时支持的图片格式。
EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".heic", ".heif", ".webp",
              ".tif", ".tiff", ".bmp"}


def image_paths(root: Path):
    """查找目录下所有支持的图片，并按路径排序以保证处理顺序稳定。"""
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def make_model(weights_path: Path, device):
    """加载本地权重，并移除 ResNet50 分类层以输出视觉特征向量。"""
    model = models.resnet50(weights=None)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state)
    # 分类层输出类别分数；替换为 Identity 后保留分类层之前的特征。
    model.fc = torch.nn.Identity()
    return model.to(device).eval()


def main():
    """解析参数、提取特征、聚类，并复制生成分组预览。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("cluster_preview"))
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    # 支持命令行中的 ~/Pictures 等用户目录写法。
    paths = image_paths(args.source.expanduser())
    if not paths:
        raise SystemExit("没有找到支持的图片文件。")
    if args.clusters < 2 or args.clusters > len(paths):
        raise SystemExit(f"聚类数必须在 2 到 {len(paths)} 之间。")

    # Apple Silicon 优先使用 MPS；其他环境回退到 CPU。
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"图片数: {len(paths)} | 设备: {device}")
    model = make_model(args.weights.expanduser(), device)
    transform = transforms.Compose([
        # 使用 ResNet50 训练时对应的尺寸和 ImageNet 归一化参数。
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 两个列表保持相同顺序，便于将聚类标签映射回原图片路径。
    features, valid_paths = [], []
    with torch.inference_mode():
        for start in tqdm(range(0, len(paths), args.batch_size), desc="提取特征"):
            batch_paths, batch_images = [], []
            for path in paths[start:start + args.batch_size]:
                try:
                    # 统一转为 RGB，兼容灰度图、RGBA 图和调色板图片。
                    with Image.open(path) as image:
                        batch_images.append(transform(image.convert("RGB")))
                    batch_paths.append(path)
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    print(f"跳过无法读取的文件: {path} ({exc})")
            if batch_images:
                batch = torch.stack(batch_images).to(device)
                output = model(batch).detach().cpu().numpy()
                # L2 归一化后，特征距离更适合比较图片的视觉相似度。
                output /= np.linalg.norm(output, axis=1, keepdims=True).clip(min=1e-12)
                features.append(output)
                valid_paths.extend(batch_paths)

    # 将各批次特征合并为矩阵，每一行对应一张有效图片。
    matrix = np.concatenate(features)
    # 固定随机种子，让相同输入下的聚类结果尽可能可复现。
    labels = KMeans(n_clusters=args.clusters, n_init=10, random_state=42).fit_predict(matrix)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path, label in zip(valid_paths, labels):
        group = args.output / f"group_{label + 1:03d}"
        group.mkdir(exist_ok=True)
        target = group / path.name
        if target.exists():
            # 不同源目录可能有同名文件，追加哈希避免覆盖已有副本。
            target = group / f"{path.stem}_{abs(hash(str(path))) % 100000}{path.suffix}"
        # copy2 会保留基本元数据，且这里只复制、不移动或删除原图。
        shutil.copy2(path, target)
        rows.append((str(path), str(group.name), str(target)))
    with (args.output / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        # 清单记录源文件、分组和预览副本的对应关系，便于追溯。
        writer.writerow(["source", "group", "preview_copy"])
        writer.writerows(rows)
    print(f"完成：已将 {len(rows)} 张图片复制到 {args.output}")


if __name__ == "__main__":
    main()
