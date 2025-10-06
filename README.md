🧠 Neural Network from Scratch (NumPy Implementation on MNIST data)

This project implements a fully-connected feedforward neural network built entirely from scratch using NumPy — without TensorFlow or PyTorch’s high-level APIs.

It trains on the MNIST handwritten digits dataset (0–9) and performs forward propagation, backward propagation, gradient checking, and cost visualization — all manually coded.


🚀 Features


✅ Implements:

ReLU and Softmax activation functions (includes other functions like sigmoid and softmax function ready-to-use)

Categorical Cross-Entropy loss

He Initialization for weights

L2 Regularization

Gradient Checking for debugging backpropagation

Training/Test Cost and Accuracy plots

Visualization of learned weights and activations


📊 Model Architecture Example


Default architecture (customizable in code):

Input layer: 784 neurons  (28×28 image)

Hidden layer 1: 25 neurons (ReLU)

Hidden layer 2: 25 neurons  (ReLU)

Output layer: 10 neurons   (Softmax)

🧩 Math Overview


Forward Propagation: z[l]=W[l]a[l−1]+b[l]

backward propagation: 

<img width="643" height="358" alt="image" src="https://github.com/user-attachments/assets/9a62330e-a0ce-42c9-81b9-a6b261cc40b3" />


🧪 Gradient Checking

To verify the correctness of backpropagation, numerical gradients are computed and compared with analytical ones:

A small difference confirms correct implementation.


📦 Requirements


Install dependencies:

pip install numpy pandas matplotlib tensorflow

(TensorFlow is only used for downloading the MNIST dataset — not for training!)


⚙️ How to Run


Clone this repository:

git clone https://github.com/diwash1340E/Handwritten-Digit-Recognition.git

cd digit_recognitiono.py



Run the script:

python digit_recognitiono.py


The program will:

Load and normalize the MNIST dataset

Initialize weights

Train the model using manual backpropagation

Display cost graphs and training/testing accuracy

Visualize first-layer learned weights



📈 Example Output

Epoch 0, Train Cost: 3.5497, Test Cost: 3.9548

Epoch 500, Train Cost: 1.0002, Test Cost: 1.7528

Epoch 1000, Train Cost: 0.7231, Test Cost: 1.4957

Epoch 1000, Train Cost: 0.7231, Test Cost: 1.4957

Epoch 1500, Train Cost: 0.6252, Test Cost: 1.4124

Epoch 2000, Train Cost: 0.5678, Test Cost: 1.3681

Epoch 2000, Train Cost: 0.5678, Test Cost: 1.3681

Epoch 2500, Train Cost: 0.5275, Test Cost: 1.3396

Epoch 3000, Train Cost: 0.4965, Test Cost: 1.3210

Epoch 3500, Train Cost: 0.4713, Test Cost: 1.3083

Epoch 4000, Train Cost: 0.4499, Test Cost: 1.2991

Epoch 4500, Train Cost: 0.4314, Test Cost: 1.2919

Final Training Accuracy: 95.63%

Final Test Accuracy: 90.30%


Time Taken to run: 4.023403358459473 minutes


Cost and weight visualizations will be shown as plots.


🧠 Key Learnings

Understanding every component of a neural network

Implementing vectorized forward/backward propagation

Importance of initialization and regularization

Verifying correctness with gradient checking
