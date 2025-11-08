import torch
import torch.nn as nn
from torchvision import models

class SqueezeNet11Med(nn.Module):
    """
    Ultra-light SqueezeNet 1.1 for grayscale medical images.
    - Adapts first conv to 1-channel with RGB-mean init.
    - Uses global pooling -> flattened vec -> MLP head.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout_rate: float = 0.4):
        super().__init__()
        # Handle both old and new torchvision APIs
        if pretrained:
            try:
                w = models.SqueezeNet1_1_Weights.DEFAULT
                self.backbone = models.squeezenet1_1(weights=w)
            except Exception:
                self.backbone = models.squeezenet1_1(pretrained=True)
        else:
            try:
                self.backbone = models.squeezenet1_1(weights=None)
            except Exception:
                self.backbone = models.squeezenet1_1(pretrained=False)

        # First conv is at features.0
        conv0 = self.backbone.features[0]
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
            self.backbone.features[0] = new_conv

        # Remove default classifier; we’ll pool features ourselves
        self.backbone.classifier = nn.Identity()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        feature_dim = 512  # SqueezeNet1.1 final feature channels

        self.classifier = nn.Sequential(
            nn.Flatten(),
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
        feats = self.backbone.features(x)     # [B, 512, H', W']
        feats = self.global_pool(feats)       # [B, 512, 1, 1]
        vec   = feats.flatten(1)              # [B, 512]
        return self.classifier(vec)

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.features(x)
        feats = self.global_pool(feats).flatten(1)
        return feats
