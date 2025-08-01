import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """A depthwise separable convolution block."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvAdapter(nn.Module):
    """Wrap an existing conv layer with a trainable depthwise separable adapter."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False
        self.adapter = DepthwiseSeparableConv(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=False,
        )
        # Initialize so the adapter has no effect at start
        nn.init.zeros_(self.adapter.depthwise.weight)
        nn.init.zeros_(self.adapter.pointwise.weight)
        if self.adapter.depthwise.bias is not None:
            nn.init.zeros_(self.adapter.depthwise.bias)
        if self.adapter.pointwise.bias is not None:
            nn.init.zeros_(self.adapter.pointwise.bias)

    def forward(self, x):
        out = self.conv(x)
        # Apply the adapter on the same input so output shapes match
        return out + self.adapter(x)


def add_conv_adapters(model: nn.Module) -> None:
    """Recursively replace Conv2d modules with ConvAdapter."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Conv2d):
            setattr(model, name, ConvAdapter(module))
        else:
            add_conv_adapters(module)