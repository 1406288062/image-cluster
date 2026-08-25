# Image Cluster

检测照片中的人脸，提取人脸特征，再按相似度自动分组；同一个人会进入同一组。

默认行为：只复制图片到 `cluster_preview/group_###/`，不会移动或删除原图。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps facenet-pytorch==2.6.0
```

## 运行

```bash
python cluster_images.py --source ~/Pictures --output ./cluster_preview
```

默认使用 CPU，第一次运行会自动下载人脸检测和识别模型。可用
`--distance-threshold` 调整严格程度：数值越小，分组越严格；默认值为 `0.35`。
每张照片默认选择最大的人脸作为主人物，多人合照建议单独处理。

也可以把 `~/Downloads` 作为输入目录。第一次建议只处理一个小目录测试。

没有检测到人脸的图片会放入 `no_face/`。支持常见 JPG、PNG、HEIC、WEBP、TIFF 等格式；HEIC 需要安装 `pillow-heif`。
