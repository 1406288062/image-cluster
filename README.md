# Image Cluster

使用本地预训练 ResNet50 提取图片特征，再用 K-Means 自动分组。

默认行为：只复制图片到 `cluster_preview/group_###/`，不会移动或删除原图。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 运行

```bash
python cluster_images.py --source ~/Pictures --output ./cluster_preview --clusters 12 --weights ./models/resnet50-11ad3fa6.pth
```

也可以把 `~/Downloads` 作为输入目录。第一次建议只处理一个小目录测试。

支持常见 JPG、PNG、HEIC、WEBP、TIFF 等格式；HEIC 需要安装 `pillow-heif`。
