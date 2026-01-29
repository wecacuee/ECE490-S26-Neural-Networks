# Video Script: Batch Normalization

**Duration:** ~12 minutes
**Topic:** Batch normalization for faster training and implicit regularization

---

## Learning Objectives

By the end of this video, students will be able to:
1. Explain the internal covariate shift problem
2. Implement batch normalization from scratch
3. Understand the difference between training and inference modes
4. Apply BatchNorm correctly in neural network architectures

---

## Script

### Introduction (1 min)

Hello everyone! Today we're going to learn about batch normalization - a technique that revolutionized deep learning when it was introduced in 2015.

[VISUAL: Training curve comparison - with and without BatchNorm]

Batch normalization allows us to:
- Train deeper networks more easily
- Use higher learning rates
- Be less sensitive to weight initialization
- Get some regularization as a bonus

Let's understand how it works!

---

### Section 1: The Problem - Internal Covariate Shift (2 min)

[VISUAL: Deep network diagram showing distributions changing at each layer]

Consider a deep network during training. As we update the weights of layer $l$, the input distribution to layer $l+1$ changes.

This is called **internal covariate shift** - the distribution of layer inputs keeps shifting as the network learns.

[VISUAL: Animation showing distribution shift over training]

Why is this a problem?
- Each layer must constantly adapt to new input distributions
- Makes it hard to choose learning rates
- Slows down training
- Can lead to vanishing or exploding activations

**The idea:** What if we normalize the inputs to each layer to have zero mean and unit variance?

[VISUAL: Show before and after normalization distributions]

This would make the input distribution stable across training!

---

### Section 2: Batch Normalization Algorithm (3 min)

[VISUAL: BatchNorm computation diagram]

For a mini-batch of activations $\{x_1, ..., x_m\}$:

**Step 1: Compute batch statistics**
$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$
$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2$$

**Step 2: Normalize**
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

The $\epsilon$ (typically $10^{-5}$) prevents division by zero.

**Step 3: Scale and shift (learnable parameters)**
$$y_i = \gamma \hat{x}_i + \beta$$

[VISUAL: Highlight gamma and beta as learnable]

Wait - why add learnable $\gamma$ and $\beta$?

If we only normalized, we'd force all layer inputs to have mean 0 and variance 1. But maybe that's not optimal for every layer!

The learnable parameters let the network undo the normalization if needed. If $\gamma = \sigma_B$ and $\beta = \mu_B$, we get back the original activations.

---

### Section 3: Implementation (2 min)

[CODE DEMO: Manual BatchNorm implementation]

```python
class BatchNorm1d:
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = torch.ones(num_features)
        self.beta = torch.zeros(num_features)

        # Running statistics for inference
        self.running_mean = torch.zeros(num_features)
        self.running_var = torch.ones(num_features)

    def forward(self, x, training=True):
        if training:
            # Compute batch statistics
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta
```

[VISUAL: Diagram showing running statistics update]

Notice we maintain **running statistics** - exponential moving averages of mean and variance across training batches.

---

### Section 4: Training vs Inference (2 min)

[VISUAL: Side-by-side comparison of training and inference behavior]

**During training:**
- Normalize using current batch statistics
- Update running statistics

**During inference:**
- Use the learned running statistics (not batch statistics)
- This gives deterministic outputs

[CODE DEMO: PyTorch train/eval mode]

```python
model.train()  # BatchNorm uses batch statistics
output = model(train_batch)

model.eval()   # BatchNorm uses running statistics
output = model(test_input)  # Works even with batch_size=1!
```

Why use running statistics at inference?
- Inference may have batch size 1
- We want deterministic outputs (same input = same output)
- Running statistics approximate population statistics

---

### Section 5: BatchNorm for CNNs (1.5 min)

[VISUAL: Conv layer feature maps with BatchNorm]

For convolutional layers, we normalize per-channel, not per-pixel:

```python
# Input: (batch, channels, height, width)
# Compute mean and variance over (batch, height, width) for each channel
nn.BatchNorm2d(num_features=64)  # 64 channels
```

[VISUAL: Show which dimensions are averaged]

Each channel gets its own $\gamma$ and $\beta$ parameters.

**Where to place BatchNorm in a conv block:**

```python
# Option 1: Conv -> BN -> ReLU (most common)
x = F.relu(self.bn(self.conv(x)))

# Option 2: Conv -> ReLU -> BN (also used)
x = self.bn(F.relu(self.conv(x)))
```

There's debate about the best order, but Conv -> BN -> ReLU is standard.

---

### Section 6: BatchNorm as Regularization (1 min)

[VISUAL: Show noise injection interpretation]

BatchNorm provides implicit regularization:

1. **Noise injection:** Batch statistics vary batch-to-batch, adding noise
2. **Prevents overfitting:** Similar effect to dropout
3. **Smoother optimization landscape:** Gradient flow is more stable

[VISUAL: Loss landscape with and without BatchNorm]

Studies show BatchNorm makes the loss landscape smoother, allowing larger learning rates.

**Note:** When using BatchNorm, you may not need dropout! Many modern architectures use BatchNorm without dropout.

---

### Section 7: Layer Normalization Alternative (1 min)

[VISUAL: Compare BatchNorm vs LayerNorm]

**Layer Normalization:** Normalize across features instead of across batch

$$\hat{x}_i = \frac{x_i - \mu_L}{\sqrt{\sigma_L^2 + \epsilon}}$$

Where $\mu_L, \sigma_L$ are computed over the feature dimension.

[VISUAL: Diagram showing LN vs BN normalization dimensions]

**When to use LayerNorm:**
- Recurrent networks (batch statistics don't make sense across time steps)
- Transformers (standard choice)
- Small batch sizes

```python
nn.LayerNorm(normalized_shape=256)  # Normalize over last dimension
```

---

### Summary (0.5 min)

[VISUAL: Summary slide]

Key points:

1. **BatchNorm normalizes** layer inputs to have zero mean and unit variance
2. **Learnable $\gamma$ and $\beta$** allow the network to undo normalization if needed
3. **Training mode:** Uses batch statistics, updates running stats
4. **Inference mode:** Uses running statistics
5. **For CNNs:** Normalize per-channel (BatchNorm2d)
6. **Benefits:** Faster training, higher learning rates, implicit regularization

Next video: ResNets and skip connections!

---

## Production Notes

- Animate the distribution of activations before and after BatchNorm
- Show real training curves comparing with/without BatchNorm
- Demonstrate the impact of learning rate with BatchNorm
- Include visualization of running statistics evolution during training
