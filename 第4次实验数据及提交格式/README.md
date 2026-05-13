# 交通标志检测 - 实验报告

## 任务概述

使用 YOLOv8 模型对交通标志进行目标检测，数据集包含 15 类标志。

## 数据集

- **训练集**: `train/images` + `train/labels`
- **验证集**: `val/images` + `val/labels`
- **测试集**: `test/images` (仅图像)
- **类别数量**: 15

### 类别列表

| ID | 类别名称 |
|----|----------|
| 0 | Green Light |
| 1 | Red Light |
| 2 | Speed Limit 10 |
| 3 | Speed Limit 100 |
| 4 | Speed Limit 110 |
| 5 | Speed Limit 120 |
| 6 | Speed Limit 20 |
| 7 | Speed Limit 30 |
| 8 | Speed Limit 40 |
| 9 | Speed Limit 50 |
| 10 | Speed Limit 60 |
| 11 | Speed Limit 70 |
| 12 | Speed Limit 80 |
| 13 | Speed Limit 90 |
| 14 | Stop |

## 模型训练

### 训练配置

**版本 1 (v2) - 初始训练**

```python
model = YOLO("yolov8n.pt")  # 从预训练权重初始化
results = model.train(
    data="data.yaml",
    epochs=10,
    imgsz=416,
    batch=16,
    device=0
)
```

**版本 2 (v3) - 继续训练**

```python
model = YOLO("runs/detect/runs/detect/traffic_signs_v2/weights/best.pt")  # 从v2最佳权重继续
results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8,
    device=0,
    lr0=0.001,
    lrf=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    warmup_epochs=3,
    patience=50,
    box=7.5,
    cls=0.5,
    dfl=1.5,
)
```

### 训练参数对比

| 参数 | v2 | v3 |
|------|----|----|
| 基础模型 | yolov8n.pt | v2 最佳权重 |
| epochs | 10 | 50 |
| imgsz | 416 | 640 |
| batch | 16 | 8 |
| patience | 100 | 50 |
| lr0 | 0.01 | 0.001 |
| 最终 mAP@0.5 | 0.879 | **0.970** |

## 训练结果

### v3 最终性能指标 (Epoch 50)

| 指标 | 数值 |
|------|------|
| Precision | 0.957 |
| Recall | 0.928 |
| **mAP@0.5** | **0.970** |
| mAP@0.5:0.95 | 0.825 |

### 损失变化

| 阶段 | Box Loss | Cls Loss | DFL Loss |
|------|----------|----------|----------|
| Epoch 1 | 0.698 | 1.038 | 1.079 |
| Epoch 25 | 0.579 | 0.646 | 0.976 |
| Epoch 50 | 0.500 | 0.304 | 0.899 |

训练过程中损失持续下降，模型收敛良好。

## 生成提交文件

使用训练好的模型对测试集进行推理：

```python
model = YOLO("runs/detect/runs/detect/traffic_signs_v3/weights/best.pt")

image_paths = sorted([p for p in os.listdir("test/images") if p.endswith(('.jpg', '.png'))])

with open("submission.csv", "w", encoding="utf-8") as f:
    f.write("image_id,class_id,x_center,y_center,width,height,confidence\n")
    for img in image_paths:
        results = model.predict(source=f"test/images/{img}", conf=0.001, verbose=False)
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x_center, y_center, width, height = box.xywhn[0].tolist()
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    f.write(f"{img},{cls},{x_center:.6f},{y_center:.6f},{width:.6f},{height:.6f},{conf:.6f}\n")
```

## 提交文件格式

CSV 包含以下列（YOLO 归一化坐标）：
- `image_id`: 图像文件名
- `class_id`: 类别 ID (0-14)
- `x_center`: 边界框中心 X 坐标 (0-1)
- `y_center`: 边界框中心 Y 坐标 (0-1)
- `width`: 边界框宽度 (0-1)
- `height`: 边界框高度 (0-1)
- `confidence`: 置信度分数

## 文件结构

```
第4次实验数据及提交格式/
├── train/
│   ├── images/          # 训练图像
│   └── labels/          # 训练标签 (YOLO格式)
├── val/
│   ├── images/          # 验证图像
│   └── labels/          # 验证标签
├── test/
│   └── images/          # 测试图像
├── data.yaml            # 数据集配置
├── baseline_infer.py    # 基线推理脚本
├── train_yolov8.py      # 训练脚本 (v3)
├── generate_submission.py
├── submission.csv       # 生成的提交文件
└── runs/detect/
    └── runs/detect/
        ├── traffic_signs_v2/  # v2 训练结果
        └── traffic_signs_v3/  # v3 训练结果
            ├── weights/best.pt
            ├── weights/last.pt
            ├── results.csv
            ├── BoxPR_curve.png
            ├── confusion_matrix.png
            └── ...
```

## 模型选择

最终使用 `runs/detect/runs/detect/traffic_signs_v3/weights/best.pt`

该模型在验证集上达到 **mAP@0.5 = 0.970**，满足高精度检测需求。