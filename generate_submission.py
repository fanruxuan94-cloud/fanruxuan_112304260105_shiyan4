if __name__ == "__main__":
    import os
    os.chdir(r"D:\机器学习\实验四\第4次实验数据及提交格式")

    from ultralytics import YOLO

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

    print("Submission saved to submission.csv")