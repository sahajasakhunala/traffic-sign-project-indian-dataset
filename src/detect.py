import os
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models

from model import TrafficSignCNN


def load_class_names(csv_path: str) -> dict[int, str]:
    """Loads class ID to class name mapping from a CSV file if available."""
    if not os.path.exists(csv_path):
        return {}

    try:
        df = pd.read_csv(csv_path)
        class_id_col, name_col = None, None

        for col in df.columns:
            col_lower = col.strip().lower()
            if col_lower in ['classid', 'class_id', 'id']:
                class_id_col = col
            elif col_lower in ['name', 'class_name', 'sign_name', 'label', 'description']:
                name_col = col

        if class_id_col and name_col:
            return {int(row[class_id_col]): str(row[name_col]).strip() for _, row in df.iterrows()}
    except Exception as e:
        print(f"[WARN] Could not parse CSV for class names: {e}")

    return {}


def main():
    parser = argparse.ArgumentParser(description="Detect and Classify Traffic Signs in an Image")
    parser.add_argument('--image_path', type=str, default="test.jpg", help="Path to input image")
    parser.add_argument('--model_type', type=str, default="resnet50", choices=['custom_cnn', 'resnet50'], help="Model architecture")
    parser.add_argument('--checkpoint_path', type=str, default="models/traffic_sign_resnet50_finetuned.pth", help="Path to checkpoint (.pth)")
    parser.add_argument('--image_size', type=int, default=128, help="Model input image size (128 for resnet50)")
    parser.add_argument('--num_classes', type=int, default=58, help="Number of traffic sign classes")
    parser.add_argument('--csv_path', type=str, default="data/Indian_Dataset/traffic_sign.csv", help="Path to traffic_sign.csv")
    parser.add_argument('--output_path', type=str, default="detected_output.jpg", help="Path to save annotated output image")
    parser.add_argument('--conf_thresh', type=float, default=0.50, help="Confidence threshold")
    parser.add_argument('--min_area', type=int, default=300, help="Minimum contour area to consider as candidate")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"[ERROR] Image not found: {args.image_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Build Model
    if args.model_type == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, args.num_classes)
    else:
        model = TrafficSignCNN(num_classes=args.num_classes)

    if not os.path.exists(args.checkpoint_path):
        print(f"[ERROR] Checkpoint file not found: {args.checkpoint_path}")
        return

    print(f"Loading weights from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))

    cleaned_state_dict = {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()

    # 2. Image Preprocessing Transform
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    class_name_map = load_class_names(args.csv_path)

    # 3. Read Image & Color Segmentation
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"[ERROR] Failed to read image: {args.image_path}")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red color masks (handles lower and upper HSV red ranges)
    mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
    mask_red = mask1 + mask2

    # Blue color mask
    mask_blue = cv2.inRange(hsv, np.array([100, 100, 50]), np.array([140, 255, 255]))

    # Yellow color mask
    mask_yellow = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255]))

    combined_mask = mask_red | mask_blue | mask_yellow

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections_found = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < args.min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Aspect ratio check (avoid ultra thin/tall blobs)
        if not (0.5 <= w / float(h) <= 2.0):
            continue

        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        # Save last candidate crop for reference
        cv2.imwrite("detected_sign.jpg", crop)

        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)

        confidence = conf.item()
        pred_class = pred.item()

        if confidence >= args.conf_thresh:
            detections_found += 1
            label = class_name_map.get(pred_class, f"Class {pred_class}")
            text = f"{label} ({confidence*100:.1f}%)"

            # Draw bounding box and label overlay
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x, y - th - 6), (x + tw + 4, y), (0, 255, 0), -1)
            cv2.putText(image, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            print(f"[DETECTED] {label} | Conf: {confidence*100:.2f}% | BBox: ({x}, {y}, {w}, {h})")

    cv2.imwrite(args.output_path, image)
    print(f"\nDetection complete. Found {detections_found} traffic sign(s).")
    print(f"Annotated output saved to: {args.output_path}")


if __name__ == "__main__":
    main()