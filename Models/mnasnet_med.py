import torch
import torch.nn as nn
from torchvision import models

class MnasNet10Med(nn.Module):
    """
    MNASNet 1.0 adapted for grayscale medical MRI.
    - Preserves (optional) ImageNet pretraining; converts stem to 1-channel via RGB-mean init.
    - Backbone returns a pooled feature vector (classifier replaced by Identity).
    - Lightweight and fast on CPU; good accuracy-per-FLOP.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout_rate: float = 0.4):
        super().__init__()

        # Load backbone with robust weights handling (new/old torchvision)
        if pretrained:
            try:
                w = models.MNASNet1_0_Weights.DEFAULT
                self.backbone = models.mnasnet1_0(weights=w)
            except Exception:
                self.backbone = models.mnasnet1_0(pretrained=True)
        else:
            try:
                self.backbone = models.mnasnet1_0(weights=None)
            except Exception:
                self.backbone = models.mnasnet1_0(pretrained=False)

        # --- Adapt first conv to 1-channel while preserving pretrained weights ---
        # In torchvision, stem is layers[0] which is usually Conv-BN-ReLU (Sequential)
        stem = self.backbone.layers[0]
        conv0 = stem[0] if isinstance(stem, nn.Sequential) else stem  # conv is usually index 0
        if isinstance(conv0, nn.Conv2d) and conv0.in_channels != 1:
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding,
                bias=conv0.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight.copy_(conv0.weight.mean(dim=1, keepdim=True))
                if conv0.bias is not None:
                    new_conv.bias.copy_(conv0.bias)
            if isinstance(stem, nn.Sequential):
                stem[0] = new_conv
            else:
                self.backbone.layers[0] = new_conv  # fallback

        # Replace the default classifier with Identity so forward() returns pooled features
        # MNASNet has 'classifier' as (Dropout, Linear) after global pool
        feature_dim = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Identity()

        # Your MLP head (mirrors your other models)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )
        self._init_head()

    def _init_head(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # pooled feature vector (B, feature_dim)
        logits = self.classifier(feats)
        return logits

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
