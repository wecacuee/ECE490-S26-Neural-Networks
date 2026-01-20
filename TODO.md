# ECE490 Neural Networks - Course TODO

## Notes

- **Quizzes**: Create in markdown format, convert to QTI using [text2qti](https://github.com/wecacuee/text2qti), then import to LMS
- **Videos**: Short (<15 min) lecture videos for each sub-topic

---

## 00-intro

**Notebooks**: ✓ Complete

**Videos**:
- [ ] Course overview and expectations
- [ ] Prerequisites review

---

## 01-py-intro

**Notebooks**: ✓ Complete (3 notebooks)

**Quiz**:
- [ ] Python basics quiz

**Videos**:
- [ ] Python basics (variables, data types, control flow)
- [ ] NumPy fundamentals
- [ ] Matplotlib visualization

---

## 011-hugging-face

**Notebooks**:
- [ ] Create Hugging Face hands-on notebook (currently only markdown docs)

**Quiz**:
- [ ] Hugging Face / pretrained models quiz

**Videos**:
- [ ] Using pretrained models from Hugging Face
- [ ] Fine-tuning basics

---

## 02-linear-models

**Notebooks**: ✓ Complete (7 notebooks)

**Quiz**:
- [ ] Linear models quiz

**Videos**:
- [ ] Linear models introduction
- [ ] Plane fitting problem
- [ ] Continuous optimization basics
- [ ] Hessians and second-order methods
- [ ] Eigenvalues and eigenvectors visualization
- [ ] Perceptron algorithm
- [ ] Probabilistic perspective on linear models

---

## 021-decision-theory

**Notebooks**:
- [ ] Create Decision Theory teaching notebook (currently only reference PDFs)

**Quiz**:
- [ ] Decision theory quiz

**Videos**:
- [ ] Decision theory fundamentals
- [ ] Connection to machine learning

---

## 03-autograd

**Notebooks**: ✓ Complete (2 notebooks)

**Quiz**:
- [ ] Automatic differentiation quiz

**Videos**:
- [ ] Automatic differentiation concepts
- [ ] Implementing autograd from scratch

---

## 05-mlp

**Notebooks**: ✓ Complete (5 notebooks)

**Quiz**:
- [ ] Multi-layer perceptrons quiz

**Videos**:
- [ ] Data, models, and learning framework
- [ ] Building a micro deep learning library
- [ ] MLP architecture and forward pass
- [ ] Backpropagation in MLPs
- [ ] Universal approximation theorem

---

## 06-pytorch

**Notebooks**: ✓ Complete (2 notebooks)

**Quiz**:
- [ ] PyTorch fundamentals quiz

**Videos**:
- [ ] NumPy to PyTorch transition
- [ ] Building MLPs in PyTorch

---

## 07-cnn

**Notebooks**: ✓ Complete (3 notebooks)

**Quiz**:
- [ ] Convolutional neural networks quiz

**Videos**:
- [ ] Convolution operation explained
- [ ] CNN architecture (conv, pool, fully-connected)
- [ ] Backprop through convolutions (conv VJP)

---

## 09-vanishing

**Notebooks**: ✓ Complete (1 notebook)

**Quiz**:
- [ ] Vanishing/exploding gradients quiz

**Videos**:
- [ ] Vanishing gradients problem
- [ ] Exploding gradients and gradient clipping

---

## 10-gpt

**Notebooks**:
- [ ] Create GPT teaching notebook (has minGPT/nanoGPT implementations but no main teaching notebook)

**Quiz**:
- [ ] GPT and LLMs quiz

**Videos**:
- [ ] Language modeling basics
- [ ] GPT architecture overview
- [ ] Training LLMs

---

## 11-regularization-resnets

**Notebooks**: ✓ Complete
- ResNets.ipynb (residual networks, BasicBlock, BottleneckBlock, He init, CIFAR-10 training)

**Quiz**:
- [ ] Regularization and ResNets quiz

**Videos**: ✓ Scripts Complete
- [x] Overfitting and regularization (video-01-overfitting-regularization.md)
- [x] Dropout (video-02-dropout.md)
- [x] Batch normalization (video-03-batch-normalization.md)
- [x] ResNet and skip connections (video-04-resnets-skip-connections.md)

---

## 12-transformers

**Notebooks**: ✓ Complete
- Transformers.ipynb (self-attention, multi-head attention, positional encoding, MiniGPT, Shakespeare training)

**Quiz**:
- [ ] Transformers quiz

**Videos**: ✓ Scripts Complete
- [x] Self-attention mechanism (video-01-self-attention.md)
- [x] Multi-head attention (video-02-multi-head-attention.md)
- [x] Transformer encoder/decoder architecture (video-03-transformer-architecture.md)
- [x] Positional encoding (video-04-positional-encoding.md)

---

## 14-media

**Notebooks**:
- [ ] Clarify scope and create content (currently only summarizing.pdf)

**Quiz**:
- [ ] TBD based on content scope

**Videos**:
- [ ] TBD based on content scope

---

### RNN/LSTM (suggested: 10-rnn)

**Notebooks**:
- [ ] Create RNN/LSTM notebook (should fit between CNNs and Transformers)

**Quiz**:
- [ ] Recurrent neural networks quiz

**Videos**:
- [ ] RNN fundamentals and backprop through time
- [ ] LSTM and GRU architectures
- [ ] Sequence-to-sequence models

### Attention Mechanisms (suggested: 11-attention)

**Notebooks**:
- [ ] Create Attention mechanism notebook (foundational for transformers)

**Quiz**:
- [ ] Attention mechanisms quiz

**Videos**:
- [ ] Attention intuition and motivation
- [ ] Scaled dot-product attention

---

## Structural Improvements

- [ ] Review myst.yml - Update to include all directories in course structure
- [ ] Add README.md - Create course overview and navigation guide
