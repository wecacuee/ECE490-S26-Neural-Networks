# Video Script: Prerequisites Review

**Duration:** ~15 minutes
**Topic:** Review of mathematical prerequisites for Neural Networks

---

## Learning Objectives

By the end of this video, students will be able to:
1. Identify the key mathematical prerequisites for the course
2. Know where to find resources to fill gaps in their knowledge
3. Understand why each prerequisite is important for neural networks
4. Complete the prerequisite homework (HW0)

---

## Script

### Introduction (1 min)

Welcome back! Before we dive into neural networks, let's make sure we have the mathematical foundations in place.

[VISUAL: Prerequisites diagram showing Linear Algebra, Calculus, Probability, Programming]

This course requires background in:
1. **Linear Algebra** (MAT 258 or MAT 262)
2. **Programming** (ECE 277 or COS 225)
3. **Probability and Statistics** (STS 235 - corequisite)
4. **Multivariable Calculus** (helpful but we'll review)

Don't worry if you're rusty - this video will refresh the key concepts and point you to resources.

---

### Section 1: Why These Prerequisites? (1.5 min)

[VISUAL: Neural network with math annotations]

Let me show you why each area matters:

**Linear Algebra:**
- Neural networks ARE matrix operations
- Input data: vectors and matrices
- Weights: matrices
- Forward pass: matrix multiplication
- Everything is linear algebra!

[VISUAL: Show y = Wx + b equation]

**Calculus:**
- Training = optimization = finding minimum of loss function
- Gradients tell us which direction to move
- Backpropagation = chain rule applied systematically

[VISUAL: Gradient descent animation]

**Probability:**
- Outputs are often probabilities
- Loss functions come from probability theory
- Understanding uncertainty and distributions

[VISUAL: Softmax output as probabilities]

**Programming:**
- We implement everything in Python
- Need comfort with functions, loops, classes
- NumPy for numerical computing

---

### Section 2: Linear Algebra Essentials (4 min)

[VISUAL: Vector diagram]

**Vectors:**
A vector is an ordered list of numbers:
$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

In neural networks, vectors represent:
- Input features (e.g., pixel values of an image)
- Hidden layer activations
- Output predictions

[VISUAL: Matrix diagram]

**Matrices:**
A matrix is a 2D array of numbers:
$$\mathbf{A} \in \mathbb{R}^{m \times n}$$
- $m$ rows, $n$ columns
- Represents transformations, weights, data batches

[VISUAL: Matrix multiplication animation]

**Matrix Multiplication:**
If $\mathbf{A}$ is $m \times n$ and $\mathbf{B}$ is $n \times p$, then $\mathbf{C} = \mathbf{AB}$ is $m \times p$.

$$C_{ij} = \sum_{k=1}^{n} A_{ik} B_{kj}$$

**Key rule:** Inner dimensions must match!
- $(m \times \mathbf{n}) \cdot (\mathbf{n} \times p) = (m \times p)$

[CODE DEMO: NumPy matrix operations]
```python
import numpy as np
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B  # Matrix multiplication
```

[VISUAL: Dot product diagram]

**Dot Product:**
$$\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^{n} a_i b_i = \mathbf{a}^\top \mathbf{b}$$

- Measures similarity between vectors
- Zero when vectors are perpendicular
- Foundation of attention mechanisms!

[VISUAL: Transpose operation]

**Transpose:**
Flips rows and columns: $(A^\top)_{ij} = A_{ji}$
- If $\mathbf{A}$ is $m \times n$, then $\mathbf{A}^\top$ is $n \times m$

---

### Section 3: Calculus Essentials (3 min)

[VISUAL: Function with minimum point]

**Why Calculus?**
Training a neural network means finding weights that minimize the loss function.

$$\mathbf{w}^* = \arg\min_{\mathbf{w}} \mathcal{L}(\mathbf{w})$$

How do we find the minimum? Use derivatives!

[VISUAL: Derivative as slope]

**Single Variable:**
The derivative tells us the slope:
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

At a minimum: $f'(x^*) = 0$

**Example:** Find minimum of $f(x) = x^2 - 6x + 33$

$$f'(x) = 2x - 6 = 0 \implies x^* = 3$$

[VISUAL: Gradient as multivariable derivative]

**Multiple Variables (Gradient):**
The gradient is a vector of partial derivatives:
$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \end{bmatrix}$$

**Key property:** The gradient points in the direction of steepest increase.

[VISUAL: Gradient descent animation]

**Gradient Descent:**
To minimize, go opposite to the gradient:
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla \mathcal{L}(\mathbf{w}_t)$$

Where $\alpha$ is the learning rate.

[VISUAL: Chain rule diagram]

**Chain Rule:**
If $y = f(g(x))$, then:
$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

**This is the foundation of backpropagation!**

---

### Section 4: Probability Essentials (3 min)

[VISUAL: Probability distribution]

**Random Variables:**
A random variable $X$ takes values with certain probabilities.

**Expectation (Mean):**
$$\mathbb{E}[X] = \sum_x x \cdot P(X=x) \quad \text{(discrete)}$$
$$\mathbb{E}[X] = \int x \cdot p(x) dx \quad \text{(continuous)}$$

**Variance:**
$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$

Measures how spread out the distribution is.

[VISUAL: PDF vs PMF vs CDF]

**Types of Distributions:**
- **PMF (Probability Mass Function):** For discrete variables, $P(X = x)$
- **PDF (Probability Density Function):** For continuous variables, $p(x)$
- **CDF (Cumulative Distribution Function):** $P(X \leq x)$

[VISUAL: Normal distribution]

**Gaussian (Normal) Distribution:**
$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

Used everywhere in neural networks:
- Weight initialization
- Noise models
- Loss functions (MSE comes from Gaussian assumption)

[VISUAL: Softmax probabilities]

**Why This Matters:**
- Neural network outputs are often probabilities
- Cross-entropy loss comes from probability theory
- Understanding uncertainty is crucial for real applications

---

### Section 5: Programming Prerequisites (2 min)

[VISUAL: Python logo and code]

**Python Fundamentals:**
You should be comfortable with:
- Variables and data types
- Control flow (if/else, loops)
- Functions and classes
- File I/O

[CODE DEMO: Basic Python]
```python
def factorial(n):
    """Compute n! recursively."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # 120
```

[VISUAL: NumPy logo]

**NumPy:**
NumPy is essential for numerical computing:
```python
import numpy as np

# Create arrays
x = np.array([1, 2, 3])
A = np.random.randn(3, 3)

# Operations
y = A @ x           # Matrix-vector multiplication
z = np.exp(x)       # Element-wise exponential
mean = np.mean(x)   # Statistics
```

**We'll use Google Colab** - no local setup needed!

---

### Section 6: Self-Assessment and Resources (1.5 min)

[VISUAL: Checklist]

**Ask yourself:**
- Can I multiply two matrices by hand?
- Do I know what a gradient is?
- Can I compute a derivative using the chain rule?
- Am I comfortable with basic probability?
- Can I write a Python function?

If you answered "no" to any of these, use these resources:

[VISUAL: Resource links]

**Linear Algebra:**
- [Zico Kolter's 30-page review](http://cs229.stanford.edu/summer2020/cs229-linalg.pdf) - Quick refresher
- [Gilbert Strang's lectures](https://www.youtube.com/playlist?list=PL49CF3715CB9EF31D) - Full course

**Probability:**
- [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability)

**Calculus:**
- [Khan Academy Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus)

**Python:**
- Our Python introduction notebooks (coming next week)

---

### Section 7: Prerequisite Homework (HW0) (1 min)

[VISUAL: HW0 overview]

The prerequisite homework tests these foundations:

**Problem 1-3:** C Programming (Autograded)
- Factorial (recursion)
- Prime numbers (loops, logic)
- Date structures (structs)

**Problem 4:** Probability (25 marks)
- Random variables, expectation, variance
- PDF vs PMF vs CDF

**Problem 5:** Multivariable Calculus (20 marks)
- Finding minima using derivatives

**Problem 6:** Linear Algebra (50 marks)
- Matrix multiplication
- Dot products
- Orthogonal matrices

[VISUAL: Submission instructions]

**Submission:**
- Go to Gradescope via UMaine credentials
- Separate submissions for autograded and manually graded

**This homework helps you identify gaps** - if you struggle, spend time with the resources before the course accelerates!

---

### Summary (0.5 min)

[VISUAL: Summary diagram]

**Key takeaways:**

1. **Linear algebra** is the language of neural networks
2. **Calculus** (especially chain rule) enables training
3. **Probability** connects to loss functions and outputs
4. **Python/NumPy** is our implementation tool

**Action items:**
1. Complete the self-assessment
2. Review weak areas using provided resources
3. Complete HW0 by the deadline
4. Come to Code Jam sessions if you need help!

[VISUAL: Encouragement message]

Don't worry if you're rusty - we'll reinforce these concepts throughout the course. The prerequisite homework is designed to help you identify and fill gaps early.

See you in the next video where we'll start with Python!

---

## Production Notes

- Include animated visualizations of matrix multiplication
- Show gradient descent converging to minimum
- Display code examples that can be run in Colab
- Provide clickable links to all resources in video description
- Consider split-screen showing math alongside code implementation
