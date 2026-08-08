import os
import argparse
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
        # Look for common column name pairs
        class_id_col = None
        name_col = None

        for col in df.columns:
            col_lower = col.strip().lower()
            if col_lower in ['classid', 'class_id', 'id']:
                class_id_col = col
            elif col_lower in ['name', 'class_name', 'sign_name', 'label', 'description']:
                name_col = col

        if class_id_col and name_col:
            mapping = {}
            for _, row in df.iterrows():
                mapping[int(row[class_id_col])] = str(row[name_col]).strip()
            return mapping
    except Exception as e:
        print(f"[WARN] Could not parse CSV for class names: {e}")

    return {}


def main():
    parser = argparse.ArgumentParser(description="Predict Traffic Sign from an Image")
    parser.add_argument('--image_path', type=str, default="test.jpg", help="Path to input image")
    parser.add_argument('--model_type', type=str, default="resnet50", choices=['custom_cnn', 'resnet50'], help="Model architecture")
    parser.add_argument('--checkpoint_path', type=str, default="models/traffic_sign_resnet50_finetuned.pth", help="Path to model checkpoint (.pth)")
    parser.add_argument('--image_size', type=int, default=128, help="Image size (128 for resnet50, 64 for custom_cnn)")
    parser.add_argument('--num_classes', type=int, default=58, help="Number of target traffic sign classes")
    parser.add_argument('--csv_path', type=str, default="data/Indian_Dataset/traffic_sign.csv", help="Path to traffic_sign.csv for class names")
    parser.add_argument('--top_k', type=int, default=5, help="Number of top predictions to print")

    args = parser.parse_args()

    # 1. Check Image Existence
    if not os.path.exists(args.image_path):
        print(f"[ERROR] Image not found at path: {args.image_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Build Model Architecture
    if args.model_type == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, args.num_classes)
    else:
        model = TrafficSignCNN(num_classes=args.num_classes)

    # 3. Load Checkpoint
    if not os.path.exists(args.checkpoint_path):
        print(f"[ERROR] Checkpoint file not found: {args.checkpoint_path}")
        return

    print(f"Loading weights from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    model.load_state_dict(cleaned_state_dict)
    model.to(device)
    model.eval()

    # 4. Preprocess Image
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    image = Image.open(args.image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    # 5. Load Class Names Mapping
    class_name_map = load_class_names(args.csv_path)

    # 6. Inference
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, k=min(args.top_k, args.num_classes))

    print("\n" + "="*60)
    print(f"  Traffic Sign Prediction ({args.model_type.upper()})")
    print("="*60)
    print(f"Image Path: {args.image_path}")

    top_class_id = top_indices[0].item()
    top_conf = top_probs[0].item() * 100
    top_name = class_name_map.get(top_class_id, f"Class {top_class_id}")

    print(f"\n🏆 Top Prediction : Class {top_class_id} — {top_name}")
    print(f"   Confidence     : {top_conf:.2f}%\n")

    print(f"Top {args.top_k} Candidates:")
    print("-" * 60)
    for rank, (idx, prob) in enumerate(zip(top_indices.tolist(), top_probs.tolist()), 1):
        name = class_name_map.get(idx, f"Class {idx}")
        print(f"  {rank}. Class {idx:2d} | {name:<35s} | {prob*100:6.2f}%")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()