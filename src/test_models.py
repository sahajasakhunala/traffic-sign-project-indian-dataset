import torch
import torch.nn as nn
import torchvision.models as tv_models
from model import TrafficSignCNN

def build_architecture(model_type: str, num_classes: int = 58, weights: str | None = None) -> nn.Module:
    """
    Factory function to instantiate models for traffic sign classification.
    Supports: custom_cnn, resnet50, efficientnet_b2, efficientnet_b3, convnext_tiny.
    """
    model_type = model_type.lower()
    
    if model_type == "custom_cnn":
        return TrafficSignCNN(num_classes=num_classes)
        
    elif model_type == "resnet50":
        w = tv_models.ResNet50_Weights.DEFAULT if weights == "imagenet" else None
        model = tv_models.resnet50(weights=w)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model
        
    elif model_type == "efficientnet_b2":
        w = tv_models.EfficientNet_B2_Weights.DEFAULT if weights == "imagenet" else None
        model = tv_models.efficientnet_b2(weights=w)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
        
    elif model_type == "efficientnet_b3":
        w = tv_models.EfficientNet_B3_Weights.DEFAULT if weights == "imagenet" else None
        model = tv_models.efficientnet_b3(weights=w)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model
        
    elif model_type == "convnext_tiny":
        w = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if weights == "imagenet" else None
        model = tv_models.convnext_tiny(weights=w)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        return model
        
    else:
        raise ValueError(f"Unsupported model type: '{model_type}'. Choose from ['custom_cnn', 'resnet50', 'efficientnet_b2', 'efficientnet_b3', 'convnext_tiny'].")


def run_smoke_test():
    """Runs forward pass smoke tests across all supported architectures."""
    batch_size = 4
    num_classes = 58
    image_size = 128
    dummy_input = torch.randn(batch_size, 3, image_size, image_size)

    candidate_models = ["custom_cnn", "resnet50", "efficientnet_b2", "efficientnet_b3", "convnext_tiny"]

    print("=" * 70)
    print("  Model Architecture Smoke Test (Target Classes: 58)")
    print("=" * 70)

    all_passed = True
    for m_type in candidate_models:
        try:
            model = build_architecture(m_type, num_classes=num_classes)
            model.eval()
            with torch.no_grad():
                out = model(dummy_input)
            
            assert out.shape == (batch_size, num_classes), f"Expected shape ({batch_size}, {num_classes}), got {out.shape}"
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  [PASS] {m_type:<18} | Logits Shape: {str(list(out.shape)):<12} | Parameters: {param_count:,}")
        except Exception as e:
            print(f"  [FAIL] {m_type:<18} | Error: {e}")
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("  ALL MODEL SMOKE TESTS PASSED SUCCESSFULLY! Ready for training.")
    else:
        print("  SOME MODEL SMOKE TESTS FAILED! Check error messages above.")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
