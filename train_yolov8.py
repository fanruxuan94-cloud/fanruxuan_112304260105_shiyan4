if __name__ == "__main__":
    import os
    os.chdir(r"D:\机器学习\实验四\第4次实验数据及提交格式")

    from ultralytics import YOLO

    # Use the existing best model as starting point
    model = YOLO("runs/detect/runs/detect/traffic_signs_v2/weights/best.pt")

    # Train for more epochs with larger image size
    results = model.train(
        data="data.yaml",
        epochs=50,           # More epochs
        imgsz=640,           # Larger image size for better accuracy
        batch=8,             # Smaller batch for larger images
        device=0,
        project="runs/detect",
        name="traffic_signs_v3",
        exist_ok=True,
        amp=False,
        workers=0,
        patience=50,         # Stop if no improvement for 50 epochs
        lr0=0.001,           # Higher learning rate for fine-tuning
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
    )