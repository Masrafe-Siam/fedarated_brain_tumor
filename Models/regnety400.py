import torch
import torch.nn as nn
from torchvision import models

class RegNetY400Med(nn.Module):
    """
    RegNetY-400MF adapted for grayscale brain MRI.
    - ImageNet pretrain optional; first conv adapted to 1-channel (RGB-mean init).
    - backbone.fc -> Identity so forward() yields pooled features into our MLP head.
    - extract_features() returns the same pooled vector for KD/XAI.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout_rate: float = 0.4):
        super().__init__()

        # Load backbone with new/old API compatibility
        if pretrained:
            try:
                w = models.RegNet_Y_400MF_Weights.DEFAULT
                self.backbone = models.regnet_y_400mf(weights=w)
            except Exception:
                self.backbone = models.regnet_y_400mf(pretrained=True)
        else:
            try:
                self.backbone = models.regnet_y_400mf(weights=None)
            except Exception:
                self.backbone = models.regnet_y_400mf(pretrained=False)

        # --- Adapt stem to 1-channel while preserving pretrained weights ---
        # RegNet stem is usually a Sequential; index 0 is Conv2d
        stem = self.backbone.stem
        conv0 = stem[0] if isinstance(stem, nn.Sequential) else getattr(stem, "conv", None)
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
                stem.conv = new_conv  # fallback if stem isn't Sequential

        # Get pooled feature dim and strip built-in classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Your MLP head (same style as your other models)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
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
        feats = self.backbone(x)  # pooled feature vector [B, num_features]
        logits = self.classifier(feats)
        return logits

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
