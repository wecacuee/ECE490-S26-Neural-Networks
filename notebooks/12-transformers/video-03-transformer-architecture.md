# Video Script: Transformer Architecture

**Duration:** ~15 minutes
**Topic:** Complete Transformer block, encoder-decoder architecture, and design choices

---

## Learning Objectives

By the end of this video, students will be able to:
1. Describe the complete Transformer block (attention + FFN + residuals)
2. Explain the difference between encoder and decoder architectures
3. Understand Pre-LN vs Post-LN normalization
4. Build a complete Transformer from components

---

## Script

### Introduction (1 min)

Welcome back! We've learned about attention and multi-head attention. Now let's put it all together into the full Transformer architecture.

[VISUAL: The famous "Attention Is All You Need" architecture diagram]

This is the original Transformer from 2017. Today we'll understand every component!

---

### Section 1: The Transformer Block (3 min)

[VISUAL: Single Transformer block diagram]

A Transformer block has two sub-layers:
1. **Multi-Head Self-Attention**
2. **Feed-Forward Network (FFN)**

Each sub-layer has:
- Residual connection (skip connection, like ResNet!)
- Layer Normalization

[VISUAL: Block equation]

$$\mathbf{x}' = \text{LayerNorm}(\mathbf{x} + \text{MultiHeadAttn}(\mathbf{x}))$$
$$\mathbf{x}'' = \text{LayerNorm}(\mathbf{x}' + \text{FFN}(\mathbf{x}'))$$

Let's break this down:

**Multi-Head Attention:** (What we learned)
- Allows each position to gather information from all positions
- Multiple heads capture different relationships

**Feed-Forward Network:**
- Two linear layers with nonlinearity between
- Applied to each position independently

$$\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x}\mathbf{W}_1 + \mathbf{b}_1)\mathbf{W}_2 + \mathbf{b}_2$$

[VISUAL: FFN dimension expansion]

Typically: $d_{ff} = 4 \times d_{model}$

If $d_{model} = 512$, then $d_{ff} = 2048$.

**Why expand then contract?**
- More expressive nonlinear transformation
- Acts as a "memory bank" for the model

**Residual Connections:**
- Enable gradient flow through deep networks
- Same principle as ResNet!

---

### Section 2: Layer Normalization (2 min)

[VISUAL: LayerNorm computation diagram]

Layer Normalization normalizes across the feature dimension:

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- $\mu, \sigma$ computed over the feature dimension (last axis)
- $\gamma, \beta$ are learnable scale and shift parameters

[VISUAL: Compare BatchNorm vs LayerNorm]

| Aspect | BatchNorm | LayerNorm |
|--------|-----------|-----------|
| Normalize over | Batch dimension | Feature dimension |
| Depends on batch | Yes | No |
| Works with batch=1 | No | Yes |

**Why LayerNorm for Transformers?**
- Batch statistics don't make sense for sequences
- Works with any batch size (including 1 for inference)
- More stable for sequence models

[CODE DEMO]
```python
# LayerNorm in PyTorch
ln = nn.LayerNorm(d_model)  # Normalizes over last dimension
x = torch.randn(batch, seq_len, d_model)
out = ln(x)  # Same shape
```

---

### Section 3: Pre-LN vs Post-LN (2 min)

[VISUAL: Side-by-side comparison]

There are two ways to arrange the components:

**Post-LN (Original Transformer):**
```
x → Attn → Add → LN → FFN → Add → LN → out
    └───────┘       └───────┘
```
$$\text{out} = \text{LN}(\mathbf{x} + \text{Attn}(\mathbf{x}))$$

**Pre-LN (GPT-2 and most modern models):**
```
x → LN → Attn → Add → LN → FFN → Add → out
         └─────┘       └─────┘
```
$$\text{out} = \mathbf{x} + \text{Attn}(\text{LN}(\mathbf{x}))$$

[VISUAL: Gradient flow comparison]

**Why Pre-LN is preferred:**
- More stable training for deep models
- Gradients flow directly through residual path
- No learning rate warmup needed

[CODE DEMO: Pre-LN block]
```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, mask=None):
        # Pre-LN: normalize before each sub-layer
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x
```

---

### Section 4: Encoder vs Decoder (3 min)

[VISUAL: Original encoder-decoder diagram]

The original Transformer has both encoder and decoder:

**Encoder (BERT-style):**
- Bidirectional attention (see all positions)
- Used for understanding/classification

**Decoder (GPT-style):**
- Causal attention (only see past positions)
- Used for generation

[VISUAL: Attention mask comparison]

```
Encoder (bidirectional):     Decoder (causal):
  1 2 3 4 5                    1 2 3 4 5
1 ✓ ✓ ✓ ✓ ✓                  1 ✓ ✗ ✗ ✗ ✗
2 ✓ ✓ ✓ ✓ ✓                  2 ✓ ✓ ✗ ✗ ✗
3 ✓ ✓ ✓ ✓ ✓                  3 ✓ ✓ ✓ ✗ ✗
4 ✓ ✓ ✓ ✓ ✓                  4 ✓ ✓ ✓ ✓ ✗
5 ✓ ✓ ✓ ✓ ✓                  5 ✓ ✓ ✓ ✓ ✓
```

**Modern architectures often use just one:**

[VISUAL: Architecture comparison table]

| Model | Type | Use Case |
|-------|------|----------|
| BERT | Encoder only | Classification, NER, QA |
| GPT | Decoder only | Text generation |
| T5 | Encoder-decoder | Translation, summarization |
| BART | Encoder-decoder | Denoising, generation |

**For sequence-to-sequence** (translation, summarization):
- Encoder processes input
- Decoder generates output, attending to encoder outputs via cross-attention

[VISUAL: Cross-attention diagram]
```
Decoder query attends to Encoder keys/values
```

---

### Section 5: Complete GPT-style Architecture (3 min)

[VISUAL: Full GPT architecture]

Let's build a complete GPT-style model:

```python
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_len):
        super().__init__()

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx):
        B, T = idx.size()

        # Embeddings
        tok = self.tok_emb(idx)           # (B, T, d_model)
        pos = self.pos_emb(torch.arange(T))  # (T, d_model)
        x = tok + pos

        # Causal mask
        mask = torch.triu(torch.ones(T, T), diagonal=1) * float('-inf')

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Output
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits
```

[VISUAL: Data flow through architecture]

```
Tokens → Embed → + Position → [Block]×N → LN → Linear → Logits
```

---

### Section 6: Design Choices and Scaling (2 min)

[VISUAL: Scaling laws graph]

**Key hyperparameters:**
- $d_{model}$: Model dimension (512 - 12288)
- $n_{heads}$: Number of attention heads (8 - 96)
- $n_{layers}$: Number of Transformer blocks (6 - 96)
- $d_{ff}$: FFN hidden dimension (usually 4 × $d_{model}$)

[VISUAL: Model size comparison]

| Model | Layers | d_model | Heads | Parameters |
|-------|--------|---------|-------|------------|
| GPT-2 Small | 12 | 768 | 12 | 117M |
| GPT-2 Large | 36 | 1280 | 20 | 774M |
| GPT-3 | 96 | 12288 | 96 | 175B |

**Scaling insights:**
- More parameters → better performance (with enough data)
- Depth and width both matter
- Attention + FFN should be roughly balanced

[VISUAL: Compute vs performance plot]

Modern research: performance improves predictably with compute!

---

### Summary (1 min)

[VISUAL: Complete architecture diagram with labels]

Key components of a Transformer:

1. **Transformer Block:** Attention + FFN + Residuals + LayerNorm
2. **Pre-LN preferred:** Normalize before attention/FFN
3. **Encoder:** Bidirectional, for understanding
4. **Decoder:** Causal, for generation
5. **Position embeddings:** Add position information
6. **Scaling:** More parameters + more data = better

Next video: Positional encoding - how do we inject position information?

---

## Production Notes

- Use animated diagrams showing data flow through the Transformer
- Show real model configurations from popular models
- Include code that can be run in Colab
- Visualize attention patterns at different layers
- Compare Pre-LN vs Post-LN training curves
