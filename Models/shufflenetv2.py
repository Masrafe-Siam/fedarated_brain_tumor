import torch
import torch.nn as nn
from torchvision import models

class ShuffleNetV2x10Med(nn.Module):
    """
    Lightweight ShuffleNetV2 (1.0x) for grayscale medical images.
    - Keeps ImageNet pretraining (if available) and adapts first conv to 1-channel.
    - Returns pooled feature vector from backbone when extract_features() is called.
    """
    def __init__(self, num_classes: int = 4, pretrained: bool = True, dropout_rate: float = 0.4):
        super().__init__()
        # Handle both old and new torchvision APIs
        self.backbone = None
        if pretrained:
            try:
                w = models.ShuffleNet_V2_X1_0_Weights.DEFAULT
                self.backbone = models.shufflenet_v2_x1_0(weights=w)
            except Exception:
                self.backbone = models.shufflenet_v2_x1_0(pretrained=True)
        else:
            try:
                self.backbone = models.shufflenet_v2_x1_0(weights=None)
            except Exception:
                self.backbone = models.shufflenet_v2_x1_0(pretrained=False)

        # Adapt first conv to 1-channel (conv1 is the stem)
        conv1 = self.backbone.conv1[0] if isinstance(self.backbone.conv1, nn.Sequential) else self.backbone.conv1
        if conv1.in_channels != 1:
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=conv1.bias is not None,
            )
            with torch.no_grad():
                new_conv.weight.copy_(conv1.weight.mean(dim=1, keepdim=True))
                if conv1.bias is not None:
                    new_conv.bias.copy_(conv1.bias)

            # Replace in stem (supports both Sequential and direct conv)
            if isinstance(self.backbone.conv1, nn.Sequential):
                self.backbone.conv1[0] = new_conv
            else:
                self.backbone.conv1 = new_conv

        # Grab feature dim and strip classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # calling backbone(x) now returns pooled features

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
        feats = self.backbone(x)     # [B, num_features]
        logits = self.classifier(feats)
        return logits

    @torch.no_grad()
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
