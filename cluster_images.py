#!/usr/bin/env python3
"""检测人脸并按人脸特征自动分组图片。"""
import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image, UnidentifiedImageError
from sklearn.cluster import DBSCAN
from tqdm import tqdm

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".heic", ".heif", ".webp", ".tif", ".tiff", ".bmp"}


def image_paths(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)


def face_embedding(image, detector, recognizer, device):
    """取照片中最大的人脸，返回归一化人脸向量；没有人脸时返回 None。"""
    boxes, probabilities = detector.detect(image)
    faces = detector(image)
    if boxes is None or faces is None:
        return None
    if faces.ndim == 3:
        faces = faces.unsqueeze(0)
    areas = []
    for left, top, right, bottom in boxes:
        areas.append(max(0.0, right - left) * max(0.0, bottom - top))
    best = int(np.argmax(areas))
    if probabilities is not None and float(probabilities[best]) < 0.90:
        return None
    with torch.inference_mode():
        vector = recognizer(faces[best].unsqueeze(0).to(device))
    vector = vector.detach().cpu().numpy()[0]
    vector /= np.linalg.norm(vector).clip(min=1e-12)
    return vector


def copy_image(path, group, output, rows, face_found):
    group_dir = output / group
    group_dir.mkdir(parents=True, exist_ok=True)
    target = group_dir / path.name
    if target.exists():
        target = group_dir / f"{path.stem}_{abs(hash(str(path))) % 100000}{path.suffix}"
    shutil.copy2(path, target)
    rows.append((str(path), group, str(target), "yes" if face_found else "no"))


def main():
    parser = argparse.ArgumentParser(description="检测人脸，并把同一个人自动复制到同一组。")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("cluster_preview"))
    parser.add_argument("--distance-threshold", type=float, default=0.35,
                        help="人脸余弦距离阈值，越小越严格；默认 0.35。")
    parser.add_argument("--min-face-size", type=int, default=40)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    paths = image_paths(args.source.expanduser())
    if not paths:
        raise SystemExit("没有找到支持的图片文件。")
    if not 0.05 <= args.distance_threshold <= 0.80:
        raise SystemExit("--distance-threshold 必须在 0.05 到 0.80 之间。")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"图片数: {len(paths)} | 设备: {device}")
    print("加载人脸模型（首次运行会自动下载模型权重）...")
    detector = MTCNN(image_size=160, margin=20, min_face_size=args.min_face_size,
                     thresholds=[0.6, 0.7, 0.7], factor=0.709, keep_all=True,
                     post_process=True, device=device)
    recognizer = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    embeddings, valid_paths, no_face_paths = [], [], []
    for path in tqdm(paths, desc="检测并提取人脸特征"):
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
            embedding = face_embedding(image, detector, recognizer, device)
            if embedding is None:
                no_face_paths.append(path)
            else:
                embeddings.append(embedding)
                valid_paths.append(path)
        except (UnidentifiedImageError, OSError, ValueError, RuntimeError) as exc:
            print(f"跳过无法读取的文件: {path} ({exc})")
            no_face_paths.append(path)

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    if embeddings:
        matrix = np.asarray(embeddings, dtype=np.float32)
        labels = DBSCAN(eps=args.distance_threshold, min_samples=1, metric="cosine").fit_predict(matrix)
        label_order = {label: i + 1 for i, label in enumerate(sorted(set(labels)))}
        for path, label in zip(valid_paths, labels):
            copy_image(path, f"group_{label_order[label]:03d}", output, rows, True)
    for path in no_face_paths:
        copy_image(path, "no_face", output, rows, False)

    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source", "group", "preview_copy", "face_detected"])
        writer.writerows(rows)
    groups = len(set(r[1] for r in rows if r[1] != "no_face"))
    print(f"完成：检测到人脸 {len(valid_paths)} 张，未检测到人脸 {len(no_face_paths)} 张")
    print(f"已生成 {groups} 个人脸分组：{output}")


if __name__ == "__main__":
    main()
