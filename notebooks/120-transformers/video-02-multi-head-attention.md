# Video Script: Multi-Head Attention

**Duration:** ~10 minutes
**Topic:** Multi-head attention and why multiple attention heads help

---

## Learning Objectives

By the end of this video, students will be able to:
1. Explain why a single attention head is limiting
2. Describe the multi-head attention architecture
3. Implement multi-head attention efficiently
4. Understand how heads learn different patterns

---

## Script

### Introduction (1 min)

Welcome back! Last time we learned about scaled dot-product attention. Today we're going to see why we want MULTIPLE attention heads running in parallel.

[VISUAL: Single head vs multi-head comparison]

A single attention head can only focus on one type of relationship at a time. But language (and other data) has many types of relationships happening simultaneously!

Let's see how multi-head attention solves this.

---

### Section 1: The Limitation of Single-Head Attention (2 min)

[VISUAL: Sentence with multiple relationships to capture]

Consider this sentence: "The cat that sat on the mat was happy."

[VISUAL: Show different relationship types]

There are multiple things to track:
1. **Syntactic:** "was" relates to "cat" (subject-verb agreement)
2. **Semantic:** "happy" relates to "cat" (who is happy?)
3. **Positional:** "mat" is near "sat" (local context)
4. **Coreference:** "that" refers to "cat"

[VISUAL: Single attention trying to capture all]

A single attention head produces ONE set of weights. It has to average across all these relationships!

$$\text{weights} = \text{softmax}(\mathbf{q}^\top \mathbf{K} / \sqrt{d_k})$$

If "was" needs to strongly attend to both "cat" and "happy" for different reasons, a single head struggles.

**Solution:** Use multiple heads, each learning different patterns!

---

### Section 2: Multi-Head Attention Architecture (2.5 min)

[VISUAL: Multi-head attention diagram]

Multi-head attention runs $h$ attention functions in parallel:

$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\mathbf{W}^O$$

Where each head is:
$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

[VISUAL: Parallel heads processing]

Each head has its own projection matrices:
- $\mathbf{W}_i^Q \in \mathbb{R}^{d_{model} \times d_k}$
- $\mathbf{W}_i^K \in \mathbb{R}^{d_{model} \times d_k}$
- $\mathbf{W}_i^V \in \mathbb{R}^{d_{model} \times d_v}$

**Key insight:** Each head operates in a lower-dimensional space!

[VISUAL: Dimension breakdown]

If $d_{model} = 512$ and $h = 8$ heads:
- Per-head dimension: $d_k = d_v = 512/8 = 64$
- Total parameters: Same as one big head!

After computing all heads, we:
1. Concatenate: $(B, T, h \times d_v)$
2. Project back: $\mathbf{W}^O \in \mathbb{R}^{h \cdot d_v \times d_{model}}$

---

### Section 3: Efficient Implementation (2.5 min)

[CODE DEMO: Multi-head attention implementation]

The naive way: loop over heads. The efficient way: batch computation!

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Combined projection for all heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.size()

        # Project to Q, K, V all at once
        qkv = self.W_qkv(x)  # (B, T, 3*C)

        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)  # Each (B, T, C)

        # Reshape for multi-head: (B, T, C) -> (B, n_heads, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention (batched over heads)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask
        weights = F.softmax(scores, dim=-1)
        out = weights @ v  # (B, n_heads, T, d_k)

        # Concatenate heads: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        return self.W_o(out)
```

[VISUAL: Tensor shape transformations]

The key trick: reshape and transpose to process all heads in parallel!

```
(B, T, d_model)
  → project → (B, T, 3*d_model)
  → split → 3 × (B, T, d_model)
  → reshape → 3 × (B, n_heads, T, d_k)
  → attention → (B, n_heads, T, d_k)
  → reshape → (B, T, d_model)
  → project → (B, T, d_model)
```

---

### Section 4: What Do Different Heads Learn? (2 min)

[VISUAL: Attention pattern visualization from BERT/GPT]

Research has shown that different heads learn different patterns:

**Head 1:** Syntactic attention
[VISUAL: Head attending to grammatical dependencies]
- Verbs attend to subjects
- Adjectives attend to nouns

**Head 2:** Positional attention
[VISUAL: Head with diagonal or nearby attention]
- Attend to previous/next token
- Local context window

**Head 3:** Semantic attention
[VISUAL: Head connecting semantically related words]
- Synonyms attend to each other
- Related concepts cluster

**Head 4:** Separator attention
[VISUAL: Head attending to punctuation/special tokens]
- Attend to sentence boundaries
- Connect to [CLS] token

[VISUAL: All heads combined]

Together, heads capture a rich set of relationships that a single head couldn't!

---

### Section 5: Design Choices (1 min)

[VISUAL: Table of common configurations]

| Model | d_model | n_heads | d_k |
|-------|---------|---------|-----|
| BERT-base | 768 | 12 | 64 |
| BERT-large | 1024 | 16 | 64 |
| GPT-2 | 768 | 12 | 64 |
| GPT-3 | 12288 | 96 | 128 |

**Common pattern:** $d_k = 64$ is standard, adjust n_heads with model size.

**Why not more heads?**
- Too many heads → each head has fewer dimensions → less capacity per head
- Typical sweet spot: 8-16 heads for medium models

[VISUAL: Head pruning research]

Research shows you can often remove many heads without hurting performance - some heads are redundant!

---

### Summary (0.5 min)

[VISUAL: Multi-head diagram with key points]

Key takeaways:

1. **Single head is limiting** - can only capture one relationship type
2. **Multi-head:** Run $h$ attention heads in parallel
3. **Same compute:** $h$ heads × $d_k$ dims = $d_{model}$ total
4. **Efficient:** Batch computation with reshape/transpose
5. **Different patterns:** Each head learns different relationships
6. **Typical config:** 8-16 heads, $d_k = 64$

Next video: The full Transformer architecture!

---

## Production Notes

- Visualize actual attention patterns from trained models (e.g., BertViz)
- Animate the parallel head computation
- Show tensor shapes at each step
- Compare single-head vs multi-head on a concrete example
- Include efficiency comparison (loop vs batched)
