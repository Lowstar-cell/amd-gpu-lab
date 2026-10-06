"""
Kleines U-Net fuer Bild -> Tiefe (die gelernte RUECKWAERTS-Richtung).

Input : (B, 3, H, W) RGB in [0,1]
Output: (B, 1, H, W) Tiefe in [0,1] (1 = nah, 0 = fern), via Sigmoid

Optional `coord=True` haengt einen normalisierten y-Koordinaten-Kanal an. Das
hilft genau dem Boden-Gradient-Cue (absolute Bildhoehe -> Tiefe), den reine
Faltungen wegen ihrer Translationsinvarianz nur schwer sehen. Echte
Tiefenmodelle nutzen denselben Trick.

Kein Dropout (ROCm/MIOpen-Bug, siehe CLAUDE.md). BatchNorm ist ok.
"""

import torch
import torch.nn as nn


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class DepthUNet(nn.Module):
    def __init__(self, base=32, coord=True):
        super().__init__()
        self.coord = coord
        cin = 3 + (1 if coord else 0)

        self.e1 = conv_block(cin, base)
        self.e2 = conv_block(base, base * 2)
        self.e3 = conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = conv_block(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = conv_block(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = conv_block(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        if self.coord:
            b, _, h, w = x.shape
            yy = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype)
            yy = yy.view(1, 1, h, 1).expand(b, 1, h, w)
            x = torch.cat([x, yy], dim=1)

        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.d3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.d2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.out(d1))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
