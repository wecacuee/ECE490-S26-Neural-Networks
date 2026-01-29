# Video Script: Positional Encoding

**Duration:** ~10 minutes
**Topic:** Why position matters and how to encode it in Transformers

---

## Learning Objectives

By the end of this video, students will be able to:
1. Explain why Transformers need positional information
2. Derive sinusoidal positional encodings
3. Compare fixed vs learned position embeddings
4. Understand relative position encodings

---

## Script

### Introduction (1 min)

Welcome to our final video on Transformers! Today we're tackling an important question: how do Transformers know where words are in a sequence?

[VISUAL: Two sentences with same words, different meaning]

"The cat ate the rat" vs "The rat ate the cat"

Same words, very different meanings! Position matters.

But here's the problem: self-attention is **permutation equivariant**. Let's understand what that means and how we fix it.

---

### Section 1: The Position Problem (2 min)

[VISUAL: Attention matrix showing permutation equivariance]

Self-attention computes:
$$\text{Attention}(\mathbf{X}) = \text{softmax}\left(\frac{\mathbf{X}\mathbf{W}^Q (\mathbf{X}\mathbf{W}^K)^\top}{\sqrt{d_k}}\right) \mathbf{X}\mathbf{W}^V$$

**Key property:** If we permute the input, the output permutes the same way!

[VISUAL: Show permutation in → permutation out]

```
Input: [A, B, C] → Output: [O_A, O_B, O_C]
Input: [C, A, B] → Output: [O_C, O_A, O_B]
```

This is nice for some problems (like sets), but bad for sequences!

**The model has no way to distinguish:**
- "dog bites man" from "man bites dog"
- "not good" from "good not"

[VISUAL: Show identical attention weights for permuted sequences]

Without position information, the model treats all orderings the same.

**Solution:** Add position information to the input!

---

### Section 2: Sinusoidal Positional Encoding (3 min)

[VISUAL: Positional encoding formula]

The original Transformer uses sinusoidal functions:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

Where:
- $pos$ = position in sequence (0, 1, 2, ...)
- $i$ = dimension index
- $d_{model}$ = model dimension

[VISUAL: 3D plot of sinusoidal encodings]

Let's understand this:

**For each dimension pair $(2i, 2i+1)$:**
- We have a sin and cos at a specific frequency
- Frequency decreases as $i$ increases

[VISUAL: Show different frequencies for different dimensions]

```
Dimension 0,1: High frequency (changes quickly with position)
Dimension 2,3: Medium frequency
...
Dimension d-2,d-1: Low frequency (changes slowly)
```

**Why sin and cos together?**

[VISUAL: Show relative position computation]

For any fixed offset $k$, we can write:
$$PE_{pos+k} = f(PE_{pos})$$

as a linear transformation! This means relative positions can be computed.

$$\begin{bmatrix} \sin(pos + k) \\ \cos(pos + k) \end{bmatrix} = \begin{bmatrix} \cos k & \sin k \\ -\sin k & \cos k \end{bmatrix} \begin{bmatrix} \sin(pos) \\ \cos(pos) \end{bmatrix}$$

The model can learn to attend to relative positions!

---

### Section 3: Implementation (2 min)

[CODE DEMO: Positional encoding implementation]

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        # Compute the div term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # Apply sin to even indices, cos to odd
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]
```

[VISUAL: Heatmap of positional encodings]

```python
pe = PositionalEncoding(d_model=64)
plt.imshow(pe.pe[0, :100, :].numpy())
plt.xlabel('Dimension')
plt.ylabel('Position')
```

You can see the different frequencies across dimensions!

---

### Section 4: Learned vs Fixed Encodings (1.5 min)

[VISUAL: Comparison table]

| Approach | Description | Used By |
|----------|-------------|---------|
| Sinusoidal (Fixed) | Deterministic, no parameters | Original Transformer |
| Learned | Trainable embedding table | GPT, BERT |
| Relative | Encode relative, not absolute position | Transformer-XL, T5 |
| RoPE | Rotary position embedding | LLaMA, GPT-NeoX |

**Learned Position Embeddings:**

```python
self.pos_emb = nn.Embedding(max_len, d_model)

def forward(self, idx):
    positions = torch.arange(len(idx))
    return self.tok_emb(idx) + self.pos_emb(positions)
```

[VISUAL: Learned vs sinusoidal comparison]

**Pros of learned:**
- Model can learn optimal encoding
- Simple to implement

**Cons of learned:**
- Can't extrapolate beyond training length
- More parameters

**Modern trend:** Fixed encodings (like RoPE) that can extrapolate to longer sequences.

---

### Section 5: Relative Position Encodings (2 min)

[VISUAL: Absolute vs relative position diagram]

**Problem with absolute position:**
- "The" at position 0 vs position 100 gets different representations
- But their relationship to neighboring words is the same!

**Relative position encodings:**
Instead of "I am at position 5", encode "I am 2 positions after you"

[VISUAL: Relative position attention matrix]

```
      pos 0  pos 1  pos 2
pos 0  [0]   [-1]   [-2]
pos 1  [+1]  [0]    [-1]
pos 2  [+2]  [+1]   [0]
```

**Rotary Position Embedding (RoPE):**

[VISUAL: Rotation in embedding space]

Key idea: Rotate query and key vectors based on position

$$\mathbf{q}'_{pos} = \mathbf{R}_{pos} \mathbf{q}$$
$$\mathbf{k}'_{pos} = \mathbf{R}_{pos} \mathbf{k}$$

Where $\mathbf{R}_{pos}$ is a rotation matrix.

**Beautiful property:**
$$(\mathbf{R}_{m} \mathbf{q})^\top (\mathbf{R}_{n} \mathbf{k}) = \mathbf{q}^\top \mathbf{R}_{m-n} \mathbf{k}$$

The attention score depends only on relative position $m-n$!

[VISUAL: Show this enables extrapolation to longer sequences]

---

### Section 6: Position Encoding in Practice (1 min)

[VISUAL: Different models' choices]

**BERT:** Learned absolute positions (max 512)
**GPT-2:** Learned absolute positions (max 1024)
**GPT-3:** Learned absolute positions (max 2048)
**LLaMA:** RoPE (can extrapolate)
**GPT-4:** Likely uses advanced relative position methods

**Current best practices:**
1. For fixed-length tasks: Learned embeddings are fine
2. For variable/long sequences: Use relative position (RoPE)
3. For efficiency: Combine with sparse attention

[VISUAL: Context length evolution]

Context lengths have grown from 512 → 100K+ tokens!

---

### Summary (0.5 min)

[VISUAL: Key points with formulas]

Key takeaways:

1. **Self-attention is position-blind** - needs explicit position info
2. **Sinusoidal encoding:** $\sin(pos/10000^{2i/d})$ and $\cos(pos/10000^{2i/d})$
3. **Learned vs fixed:** Trade-off between flexibility and extrapolation
4. **Relative position:** Encode offset, not absolute position
5. **RoPE:** Rotation-based, enables length extrapolation

You now have all the pieces to understand and build Transformers!

---

## Production Notes

- Visualize sinusoidal patterns at different frequencies
- Animate the rotation in RoPE
- Show extrapolation failure for learned embeddings
- Compare attention patterns with different position encodings
- Include code for both sinusoidal and learned embeddings
