# Video Script: Self-Attention Mechanism

**Duration:** ~12 minutes
**Topic:** Understanding attention as soft dictionary lookup and the self-attention mechanism

---

## Learning Objectives

By the end of this video, students will be able to:
1. Explain attention as a soft dictionary lookup
2. Understand the Query, Key, Value formulation
3. Implement scaled dot-product attention
4. Explain why we scale by √d_k

---

## Script

### Introduction (1 min)

Welcome! Today we're going to explore one of the most important ideas in modern deep learning: the attention mechanism.

[VISUAL: Show various applications - GPT, BERT, image models, protein folding]

Attention is the foundation of Transformers, which power:
- Large language models like GPT and Claude
- BERT for understanding text
- Vision Transformers for images
- AlphaFold for protein structure prediction

Let's understand how it works from first principles.

---

### Section 1: The Intuition - Soft Dictionary Lookup (2.5 min)

[VISUAL: Traditional dictionary lookup diagram]

Think about a dictionary - a collection of key-value pairs:
- "apple" → "a red fruit"
- "banana" → "a yellow fruit"
- "cherry" → "a small red fruit"

Given a query "red fruit", we find the matching key and return its value.

[VISUAL: Show exact match returning "apple" entry]

But what if there's no exact match? Traditional lookup fails.

**Attention provides a "soft" lookup:**

[VISUAL: Soft lookup diagram with weighted average]

Instead of finding one exact match, attention returns a **weighted combination** of all values, where weights depend on how similar each key is to the query.

$$\text{Attention}(\mathbf{q}, \mathbf{K}, \mathbf{V}) = \sum_{i=1}^{n} \alpha_i \mathbf{v}_i$$

Where the attention weights $\alpha_i$ measure similarity:

$$\alpha_i = \frac{\exp(\mathbf{q}^\top \mathbf{k}_i)}{\sum_j \exp(\mathbf{q}^\top \mathbf{k}_j)} = \text{softmax}(\mathbf{q}^\top \mathbf{K})_i$$

[VISUAL: Animation showing query "red fruit" producing high weights for "apple" and "cherry"]

If the query is "red fruit", it gets high weights for "apple" and "cherry", producing a blended output!

---

### Section 2: Query, Key, Value Explained (2 min)

[VISUAL: Q, K, V diagram with analogy]

Let's make these terms concrete:

**Query (Q):** "What am I looking for?"
- The question you're asking
- The search term

**Key (K):** "What information do I have?"
- Labels or descriptors for stored information
- Like tags on a filing system

**Value (V):** "What should I return?"
- The actual content to retrieve
- The information you want

[VISUAL: Library analogy]

Think of a library:
- **Query:** "I want books about machine learning"
- **Keys:** Subject tags on books ("machine learning", "history", "cooking")
- **Values:** The actual book contents

Attention finds books with similar tags to your query and gives you a weighted mix of their contents.

**The insight:** Q, K, V are often derived from the same input through learned projections!

---

### Section 3: Scaled Dot-Product Attention (3 min)

[VISUAL: Full attention equation]

The standard attention formula in Transformers:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right) \mathbf{V}$$

Let's break this down:

[VISUAL: Step-by-step computation]

**Step 1: Compute attention scores**
$$\text{scores} = \mathbf{Q}\mathbf{K}^\top$$

This is a matrix of dot products between every query and every key.
Shape: (n_queries, n_keys)

**Step 2: Scale by √d_k**
$$\text{scaled\_scores} = \frac{\text{scores}}{\sqrt{d_k}}$$

We'll explain why shortly!

**Step 3: Apply softmax**
$$\text{weights} = \text{softmax}(\text{scaled\_scores})$$

Each row sums to 1 - it's a probability distribution over keys.

**Step 4: Weighted sum of values**
$$\text{output} = \text{weights} \cdot \mathbf{V}$$

[CODE DEMO: Python implementation]

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.size(-1)

    # Step 1 & 2: Compute and scale scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Optional: apply mask
    if mask is not None:
        scores = scores + mask  # mask has -inf for blocked positions

    # Step 3: Softmax
    weights = F.softmax(scores, dim=-1)

    # Step 4: Weighted sum
    output = torch.matmul(weights, V)

    return output, weights
```

---

### Section 4: Why Scale by √d_k? (2 min)

[VISUAL: Distribution of dot products]

This is a subtle but important detail!

If query and key elements are standard normal: $q_i, k_i \sim \mathcal{N}(0, 1)$

Then the dot product has:
$$\mathbb{E}[\mathbf{q}^\top \mathbf{k}] = 0$$
$$\text{Var}(\mathbf{q}^\top \mathbf{k}) = d_k$$

[VISUAL: Show variance growing with d_k]

For large $d_k$, dot products have large magnitude!

**Problem:** Softmax saturates for large inputs.

[VISUAL: Softmax curve showing saturation]

When inputs are large (say, ±100), softmax outputs are nearly 0 or 1. Gradients become tiny!

$$\text{softmax}([100, 0, 0]) \approx [1, 0, 0]$$

**Solution:** Scale by $\sqrt{d_k}$ to normalize variance back to 1.

$$\text{Var}\left(\frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1$$

[VISUAL: Before and after scaling showing healthier softmax outputs]

Now softmax operates in a good regime where gradients flow!

---

### Section 5: Self-Attention (2 min)

[VISUAL: Self-attention diagram]

In **self-attention**, Q, K, and V all come from the same sequence!

Given input sequence $\mathbf{X} \in \mathbb{R}^{n \times d}$:

$$\mathbf{Q} = \mathbf{X}\mathbf{W}^Q$$
$$\mathbf{K} = \mathbf{X}\mathbf{W}^K$$
$$\mathbf{V} = \mathbf{X}\mathbf{W}^V$$

Where $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V$ are learned projection matrices.

[VISUAL: Show input sentence being projected to Q, K, V]

**What does self-attention do?**

Each position in the sequence "attends" to all other positions to gather relevant information.

[VISUAL: Animation of "The cat sat on the mat" with attention weights]

For the word "sat":
- It might attend strongly to "cat" (who is doing the action)
- And to "mat" (where the action happens)

**Self-attention learns which parts of the input are relevant to each position.**

---

### Section 6: Visualizing Attention (1 min)

[VISUAL: Attention heatmap for a sentence]

Let's look at real attention patterns:

```
Query: "it"
         The  cat  sat  on   the  mat  .
Weights: 0.05 0.35 0.15 0.05 0.05 0.30 0.05
```

The word "it" attends most to "cat" and "mat" - the nouns it might refer to!

[VISUAL: Multiple attention patterns showing different relationships]

Different parts of the network learn different patterns:
- Some heads learn syntactic relationships
- Some learn semantic relationships
- Some learn positional patterns

---

### Summary (0.5 min)

[VISUAL: Key equations]

Today we learned:

1. **Attention is soft lookup** - weighted average based on similarity
2. **Q, K, V framework** - Query (search), Key (labels), Value (content)
3. **Scaled dot-product:** $\text{softmax}(\mathbf{QK}^\top/\sqrt{d_k})\mathbf{V}$
4. **Scale by √d_k** - prevents softmax saturation
5. **Self-attention** - Q, K, V derived from the same input

Next video: Multi-head attention - running multiple attention computations in parallel!

---

## Production Notes

- Create interactive visualization of attention weights
- Animate the soft lookup concept
- Show real attention patterns from a trained model
- Code demos should be runnable in Colab
- Use color coding consistently for Q (blue), K (green), V (orange)
