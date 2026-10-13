
import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4
DROPOUT = 0.3


def build_model(
    num_classes=NUM_CLASSES,
    pretrained=True,
    freeze_backbone=True
):
    # Load pretrained EfficientNet-B2
    weights = (
        models.EfficientNet_B2_Weights.DEFAULT
        if pretrained else None
    )

    model = models.efficientnet_b2(weights=weights)

    # Freeze the feature extractor initially
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace the original classifier
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT),
        nn.Linear(in_features, num_classes)
    )

    return model


def unfreeze_last_blocks(model):
    # Fine-tune the last two feature blocks
    for param in model.features[-2:].parameters():
        param.requires_grad = True


if __name__ == "__main__":
    model = build_model()

    print("Model loaded successfully!")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Trainable:", name)

    dummy_input = torch.randn(2, 3, 260, 260)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print("Output shape:", output.shape)
