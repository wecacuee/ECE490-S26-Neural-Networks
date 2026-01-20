# Video Script: ResNets and Skip Connections

**Duration:** ~15 minutes
**Topic:** Residual networks, skip connections, and why they enable training very deep networks

---

## Learning Objectives

By the end of this video, students will be able to:
1. Explain the degradation problem in deep networks
2. Describe the residual learning formulation
3. Understand how skip connections improve gradient flow
4. Implement BasicBlock and BottleneckBlock architectures
5. Apply He initialization for ReLU networks

---

## Script

### Introduction (1 min)

Welcome! Today we're going to talk about one of the most important architectural innovations in deep learning: Residual Networks, or ResNets.

[VISUAL: ImageNet accuracy over the years, highlight ResNet breakthrough in 2015]

In 2015, ResNet won the ImageNet challenge with a 152-layer network - dramatically deeper than anything before. The key insight was surprisingly simple: skip connections.

Let's understand why we need them and how they work.

---

### Section 1: The Degradation Problem (2.5 min)

[VISUAL: Training curve showing 56-layer plain network vs 20-layer]

Here's something surprising. If we train a 20-layer network and a 56-layer network on the same task:

**The deeper 56-layer network has HIGHER training error!**

Wait, shouldn't more layers mean more capacity? More capacity should mean we can fit the training data better, right?

[VISUAL: Diagram showing that this is NOT overfitting]

This is NOT overfitting - the training error is higher, not just the test error.

**The problem:** Deeper networks are harder to optimize. The extra layers aren't learning useful features; they're struggling to learn even the identity function!

[VISUAL: Theoretical argument diagram]

Think about it: if the extra layers just learned the identity mapping ($y = x$), the 56-layer network would be at least as good as the 20-layer network. But in practice, it's hard to learn identity with nonlinear layers.

**This is called the degradation problem.**

---

### Section 2: Residual Learning - The Key Insight (3 min)

[VISUAL: Standard block vs Residual block diagram]

Here's the brilliant insight from ResNet:

Instead of trying to learn a direct mapping $\mathcal{H}(\mathbf{x})$, let's learn the **residual**:
$$\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$$

Then the output is:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$$

[VISUAL: Animated diagram showing skip connection]

```
x ─────────────────────(+)──→ y
  │                     ↑
  └──→ [Conv] → [BN] → [ReLU] → [Conv] → [BN] ─┘
              F(x) "residual"
```

**Why is this easier?**

If the identity mapping is optimal (the extra layer should do nothing), then:
- Learning $\mathcal{H}(\mathbf{x}) = \mathbf{x}$ directly requires learning weights for identity
- Learning $\mathcal{F}(\mathbf{x}) = 0$ just requires pushing weights toward zero

[VISUAL: Show weight distribution learning toward zero]

Pushing weights toward zero is much easier! It happens naturally with weight decay regularization.

**The skip connection provides a "default" of identity** - the network only needs to learn the deviation from identity.

---

### Section 3: Gradient Flow Analysis (2.5 min)

[VISUAL: Gradient flow diagram through residual blocks]

Another crucial benefit: skip connections improve gradient flow during backpropagation.

For a residual block $\mathbf{y} = \mathcal{F}(\mathbf{x}) + \mathbf{x}$, the gradient is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \left(1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}\right)$$

[VISUAL: Highlight the "+1" term]

See that **+1** term? It means gradients always have a direct path through the skip connection!

For a network with $L$ residual blocks:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L} \cdot \prod_{l=1}^{L} \left(1 + \frac{\partial \mathcal{F}_l}{\partial \mathbf{x}_{l-1}}\right)$$

[VISUAL: Expanding the product showing identity terms]

Even if some $\frac{\partial \mathcal{F}_l}{\partial \mathbf{x}}$ terms are small, the gradient never completely vanishes because of the +1 terms.

**Skip connections create a "gradient highway"** from loss to early layers.

---

### Section 4: Basic Residual Block (2 min)

[VISUAL: BasicBlock architecture diagram]

Let's implement the basic residual block used in ResNet-18 and ResNet-34:

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample  # For dimension matching

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Dimension matching for skip connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # THE SKIP CONNECTION
        out = F.relu(out)
        return out
```

[VISUAL: Show dimension matching with 1x1 conv]

When stride > 1 or channels change, we use a 1x1 convolution to match dimensions:

```python
downsample = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=1, stride=2, bias=False),
    nn.BatchNorm2d(128)
)
```

---

### Section 5: Bottleneck Block (1.5 min)

[VISUAL: Bottleneck architecture diagram]

For deeper networks (ResNet-50+), we use **bottleneck blocks** to reduce computation:

```
x → Conv1x1 → BN → ReLU → Conv3x3 → BN → ReLU → Conv1x1 → BN → (+) → ReLU → out
    (reduce)                (process)              (expand)         ↑
    ───────────────────────────────────────────────────────────────┘
```

[CODE DEMO: Bottleneck implementation]

```python
class Bottleneck(nn.Module):
    expansion = 4  # Output channels = out_channels * 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        # ... BatchNorm layers
```

The 1x1 convolutions "squeeze" and "expand" channels, reducing the cost of the expensive 3x3 convolution.

---

### Section 6: He Initialization (1.5 min)

[VISUAL: Variance propagation diagram]

For deep networks with ReLU, we need proper weight initialization.

**Problem:** If $y = Wx$ and weights are too large or too small, activations explode or vanish.

**Analysis:** For ReLU, half the values become zero. To maintain variance:

$$\text{Var}(W_{ij}) = \frac{2}{n_{in}}$$

This is **He initialization** (after Kaiming He, who proposed ResNet):

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{in}}}\right)$$

[CODE DEMO: He initialization in PyTorch]

```python
# PyTorch does this by default for Conv2d with ReLU
nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
```

---

### Section 7: ResNet Architectures (1.5 min)

[VISUAL: Table of ResNet variants]

| Model | Blocks | Parameters |
|-------|--------|------------|
| ResNet-18 | [2, 2, 2, 2] BasicBlock | 11M |
| ResNet-34 | [3, 4, 6, 3] BasicBlock | 21M |
| ResNet-50 | [3, 4, 6, 3] Bottleneck | 25M |
| ResNet-101 | [3, 4, 23, 3] Bottleneck | 44M |
| ResNet-152 | [3, 8, 36, 3] Bottleneck | 60M |

[VISUAL: Full ResNet-18 architecture diagram]

```
Input → Conv7x7 → MaxPool →
[Stage 1: 2 blocks, 64ch] →
[Stage 2: 2 blocks, 128ch] →
[Stage 3: 2 blocks, 256ch] →
[Stage 4: 2 blocks, 512ch] →
AvgPool → FC → Output
```

---

### Section 8: Impact and Extensions (1 min)

[VISUAL: Timeline of ResNet influence]

ResNet's influence has been enormous:
- **DenseNet:** Concatenate instead of add (even more connections)
- **U-Net:** Skip connections in encoder-decoder for segmentation
- **Transformers:** Residual connections around attention layers
- **Modern architectures:** Nearly all use skip connections

[VISUAL: Show loss landscape comparison]

Research shows skip connections make the loss landscape **smoother**, allowing larger learning rates and more stable training.

---

### Summary (0.5 min)

[VISUAL: Key equations and diagrams]

Key takeaways:

1. **Degradation problem:** Deeper networks can have higher training error
2. **Residual learning:** Learn $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$, output $\mathcal{F}(\mathbf{x}) + \mathbf{x}$
3. **Skip connections:** Provide gradient highway through identity path
4. **BasicBlock:** Two 3x3 convs for ResNet-18/34
5. **Bottleneck:** 1x1-3x3-1x1 for deeper ResNets
6. **He initialization:** Use $\sqrt{2/n_{in}}$ variance for ReLU

Try implementing your own ResNet in the notebook exercises!

---

## Production Notes

- Animate the gradient flow comparison between plain and residual networks
- Show training curves for different depths with and without skip connections
- Include CIFAR-10 or ImageNet examples showing ResNet performance
- Demonstrate loss landscape visualization if possible
- Code demos should be runnable and show actual training improvement
