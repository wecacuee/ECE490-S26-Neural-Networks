# Video Script: Dropout

**Duration:** ~10 minutes
**Topic:** Dropout regularization and its interpretation as implicit ensemble

---

## Learning Objectives

By the end of this video, students will be able to:
1. Explain how dropout works during training and inference
2. Implement inverted dropout
3. Understand dropout as an implicit ensemble method
4. Choose appropriate dropout rates for different layers

---

## Script

### Introduction (1 min)

Welcome back! Today we're going to learn about dropout - one of the most important regularization techniques specifically designed for neural networks.

[VISUAL: Network diagram with some neurons "dropped out" (grayed)]

The idea is deceptively simple: during training, randomly "turn off" some neurons. But as we'll see, this simple trick has profound effects on how the network learns.

---

### Section 1: How Dropout Works (2.5 min)

[VISUAL: Animated network showing dropout in action across multiple forward passes]

Here's the basic algorithm:

**During training:**
1. For each forward pass, randomly set each neuron's output to zero with probability $p$
2. The dropped neurons change each batch - different "random subnetwork" every time
3. Remaining activations are scaled up by $\frac{1}{1-p}$ to maintain expected values

$$\tilde{h}_i = \begin{cases}
0 & \text{with probability } p \\
\frac{h_i}{1-p} & \text{with probability } 1-p
\end{cases}$$

[CODE DEMO: Manual dropout implementation]

```python
def dropout(x, p=0.5, training=True):
    if not training:
        return x
    # Create random mask
    mask = (torch.rand_like(x) > p).float()
    # Scale by 1/(1-p) to maintain expected value
    return x * mask / (1 - p)
```

**During inference:**
- Use ALL neurons (no dropout)
- No scaling needed because we already scaled during training

This is called "inverted dropout" - it's the standard implementation.

---

### Section 2: Why Does Dropout Help? (2.5 min)

[VISUAL: Diagram showing co-adaptation of neurons]

**Preventing Co-adaptation:**

Without dropout, neurons can become "lazy" - they rely on specific other neurons to fix their mistakes. This creates fragile dependencies.

[VISUAL: Show two networks - one with co-adapted neurons, one without]

With dropout, each neuron must learn to be useful on its own, because it can't rely on any specific other neuron being present.

**The Ensemble Interpretation:**

[VISUAL: Show exponentially many subnetworks]

Here's a beautiful way to think about dropout:

A network with $n$ neurons has $2^n$ possible subnetworks (each neuron is either on or off).

During training, we're effectively training all $2^n$ subnetworks simultaneously!

At test time, we use all neurons - this approximates averaging the predictions of all subnetworks.

[VISUAL: Show ensemble averaging effect]

$$\text{Ensemble Output} \approx \frac{1}{2^n} \sum_{\text{all subnetworks}} f_{\text{subnetwork}}(x)$$

This is why dropout is so powerful - it gives us the benefits of model ensembling without training multiple models!

---

### Section 3: Dropout in Practice (2 min)

[VISUAL: Network architecture diagram showing dropout placement]

**Where to apply dropout:**
- After activation functions in hidden layers
- NOT after the final output layer
- Common rates: 0.2-0.5 for fully connected layers, 0.1-0.3 for convolutional layers

[CODE DEMO: PyTorch dropout in a network]

```python
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after activation
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No dropout before output
        return x
```

**Important:** Always remember to set `model.train()` during training and `model.eval()` during inference!

```python
model.train()  # Dropout active
train(...)

model.eval()   # Dropout inactive
evaluate(...)
```

---

### Section 4: Dropout Rate Selection (1.5 min)

[VISUAL: Graph showing effect of different dropout rates]

The dropout rate $p$ is a hyperparameter:

- **p = 0**: No regularization
- **p = 0.5**: Classic dropout (common for fully connected layers)
- **p > 0.5**: Aggressive regularization (rarely used)

[VISUAL: Table of recommended rates]

| Layer Type | Typical Dropout Rate |
|------------|---------------------|
| Input layer | 0.0 - 0.2 |
| Hidden fully-connected | 0.4 - 0.5 |
| Convolutional | 0.1 - 0.3 |
| Recurrent (between layers) | 0.2 - 0.5 |

**Rule of thumb:** Start with p=0.5 for fully connected layers, then tune based on validation performance.

---

### Section 5: Variants and Alternatives (1.5 min)

[VISUAL: Comparison of dropout variants]

**DropConnect:** Drop individual weights instead of neurons
- Even more random subnetworks
- Slightly more computation

**Spatial Dropout:** For CNNs, drop entire feature maps
- Preserves spatial structure better than regular dropout

```python
# Spatial dropout in PyTorch
nn.Dropout2d(p=0.2)  # Drops entire channels
```

**DropBlock:** For CNNs, drop contiguous regions
- Prevents network from relying on any spatial location

[VISUAL: Show DropBlock regions on feature map]

**Dropout with BatchNorm:**
- Use dropout carefully with batch normalization
- Some recommend placing dropout after BN, others recommend not using both

---

### Summary (1 min)

[VISUAL: Summary slide]

Key takeaways:

1. **Dropout randomly zeros neurons** during training with probability $p$
2. **Inverted dropout** scales remaining activations by $\frac{1}{1-p}$
3. **No dropout at test time** - use all neurons
4. **Implicit ensemble** of $2^n$ subnetworks
5. **Prevents co-adaptation** - neurons learn robust features
6. **Common rates:** 0.5 for FC layers, 0.1-0.3 for conv layers

In the next video, we'll look at batch normalization - another technique that provides both regularization and faster training!

---

## Production Notes

- Animate the dropout process showing different neurons being dropped in different batches
- Show the same input producing different outputs during training vs inference
- Include accuracy comparison with and without dropout on MNIST/CIFAR
- Demonstrate the model.train() vs model.eval() behavior
