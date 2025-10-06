# 🧠 Neural Network from Scratch (NumPy Implementation on MNIST data)

This project implements a fully-connected feedforward neural network built entirely from scratch using **NumPy** — without TensorFlow or PyTorch’s high-level APIs.

It trains on the **MNIST handwritten digits dataset (0–9)** and performs **forward propagation**, **backward propagation**, **gradient checking**, and **cost visualization** — all manually coded.

---

## 🚀 Features

✅ Implements:
- **ReLU** and **Softmax** activation functions  
  (includes other functions like **Sigmoid** and **Softmax** ready-to-use)
- **Categorical Cross-Entropy** loss
- **He Initialization** for weights
- **L2 Regularization**
- **Gradient Checking** for debugging backpropagation
- **Training/Test Cost and Accuracy plots**
- **Visualization of learned weights and activations**

---

## 📊 Model Architecture Example

Default architecture (customizable in code):

| Layer | Type | Neurons | Activation |
|:------|:------|:---------|:------------|
| Input | Input Layer | 784 (28×28 image) | — |
| Hidden 1 | Dense | 25 | ReLU |
| Hidden 2 | Dense | 25 | ReLU |
| Output | Dense | 10 | Softmax |

---

### **Notation**

| Symbol | In Code | Description |
|:--------|:---------|:-------------|
| \( a[0] \) | `x` | Input data matrix |
| \( a[l] \) | `a[l]` | Activation at layer `l` |
| \( z[l] \) | `z[l]` | Linear pre-activation values |
| \( W[l] \) | `W[l]` | Weights for layer `l` |
| \( b[l] \) | `b[l]` | Biases for layer `l` |
| \( dz[l] \) | `dz[l]` | Error term at layer `l` |
| \( dW[l] \) | `dW[l]` | Gradient of weights |
| \( db[l] \) | `db[l]` | Gradient of biases |
| \( m \) | `m = x.shape[1]` | Number of training examples |
| \( \lambda \) | `lam` | Regularization constant |
| \( \alpha \) | `alpha` | Learning rate |
| \( J \) | `cost` | Cost function value |

---

## 🧪 Gradient Checking

To verify the correctness of backpropagation, numerical gradients are computed and compared with analytical ones.

A **small difference** confirms correct implementation.

---

## 📦 Requirements


Install dependencies:

```bash
pip install numpy pandas matplotlib tensorflow
```

Clone this repository and run the script:

```bash
git clone https://github.com/diwash1340E/Handwritten-Digit-Recognition.git
cd Handwritten-Digit-Recognition
python digit_recognitiono.py
```

The program will:

- **Load and normalize the MNIST dataset**
- **Initialize weights**
- **Train the model using manual backpropagation**
- **Display cost graphs and training/testing accuracy**
- **Visualize first-layer learned weights**
