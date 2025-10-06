#--Network according to the paper

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist #for data
import time

np.random.seed(42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

#Softmax activation
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))  # Subtract max for stability
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

#Softmax activation derivative
def softmax_derivative(z):
    s = softmax(z)
    return s * ( 1 - s ) 

# ReLU Activation
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU
def relu_derivative(z):
    return (z > 0).astype(float)

def final_activation(z):
    return softmax(z)
    
def final_act_derivative(z):
    return softmax_derivative(z)
    
def activation(z):
    return relu(z)
    
def activation_derivative(z):
    return relu_derivative(z)

def cost_func(y,y_hat,W,lam):
    m = y.shape[1]
    J = (1/m) * np.sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) + (lam / (2 * m)) * sum(np.sum(W[k]**2) for k in W)    
    return J

def cost_gradient(y,y_hat):
    dJ = ( (y_hat-y)/(y_hat * (1-y_hat)) )
    return dJ
    
def init(nl,layers,x): #initialzation 
    W = {}
    b = {}
    
    for i in range(1, nl):
        #Xavier initialization for Sigmoid
        '''
        epsilon_init = np.sqrt(6/ (layers[i] + layers[i-1]))
        W[i] = np.random.randn(layers[i],layers[i-1]) * 2 * epsilon_init - epsilon_init #Xavier Initialization 
        b[i] = np.zeros((layers[i],1)) 
        '''
        #He initialization for ReLU
        he_scale = np.sqrt(2 / layers[i-1])
        W[i] = np.random.randn(layers[i], layers[i-1]) * he_scale
        b[i] = np.zeros((layers[i], 1))
        
    return W,b

def one_hot(y,layers,l):
    f = np.zeros((len(y),layers[l-1]))
    for n in range(len(y)):
        loc = y[n]
        f[n,loc] = 1
    return f

def predict(W, b, x, layers):
    a = {0: x}
    z = {}
    nl = len(layers)
    for i in range(1, nl-1):
        z[i] = np.matmul(W[i], a[i-1]) + b[i]
        a[i] = activation(z[i])
        
    z[nl-1] = np.matmul(W[nl-1], a[nl-2]) + b[nl-1]
    a[nl-1] = final_activation(z[nl-1])
    return np.argmax(a[nl-1], axis=0)

def forward_prop(W, a, b, z):
    nl = len(layers)
    
    for i in range(1, nl-1):
        z[i] = np.matmul(W[i], a[i-1]) + b[i]
        a[i] = activation(z[i])
        
    z[nl-1] = np.matmul(W[nl-1], a[nl-2]) + b[nl-1]
    a[nl-1] = final_activation(z[nl-1])
    return z, a

def backward_prop(a, y, W, b, z, layers,m):
    dW = {}
    db = {}
    dz = {}
    L = len(layers) - 1

    dz[L] = cost_gradient(y, a[L]) * final_act_derivative(z[L])
    dW[L] = (1/m) * (np.matmul(dz[L],a[L-1].T))
    db[L] = (1/m) * (np.sum(dz[L], axis=1, keepdims=True))

    for l in range(L-1, 0, -1):
        dz[l] = np.matmul(W[l+1].T, dz[l+1]) * activation_derivative(z[l])
        dW[l] = (1/m) * np.matmul(dz[l], a[l-1].T)
        db[l] = (1/m) * np.sum(dz[l], axis=1, keepdims=True)
  
    return dW, db

def grad_check(x, y, W, b, layers, lam):
    # Use a small subset of data
    num_samples = 100
    x_small = x[:, :num_samples]
    y_small = y[:, :num_samples]
    a = {0: x_small}
    m = x_small.shape[1]
    
    epsilon = 1e-4
    z, a = forward_prop(W, a, b, {})
    dW, _ = backward_prop(a, y_small, W, b, z, layers, m)
    
    check_layer = 1
    numerical_grad_W = np.zeros_like(W[check_layer])
    W_original = W[check_layer].copy()
    
    # Check a subset of weights
    num_checks = 300
    indices = np.random.choice(W[check_layer].size, num_checks, replace=False)
    rows, cols = np.unravel_index(indices, W[check_layer].shape)
    
    for idx in range(num_checks):
        i, j = rows[idx], cols[idx]
        
        W[check_layer][i, j] = W_original[i, j] + epsilon
        _, a_plus = forward_prop(W, {0: x_small}, b, {})
        cost_plus = cost_func(y_small, a_plus[len(layers)-1], W, lam)

        W[check_layer][i, j] = W_original[i, j] - epsilon
        _, a_minus = forward_prop(W, {0: x_small}, b, {})
        cost_minus = cost_func(y_small, a_minus[len(layers)-1], W, lam)

        numerical_grad_W[i, j] = (cost_plus - cost_minus) / (2 * epsilon)
        W[check_layer][i, j] = W_original[i, j]  # restore weight

    # Compare only checked subset
    analytical_subset = dW[check_layer][rows, cols]
    numerical_subset = numerical_grad_W[rows, cols]

    numerator = np.linalg.norm(analytical_subset - numerical_subset)
    denominator = np.linalg.norm(analytical_subset) + np.linalg.norm(numerical_subset)
    difference = numerator / denominator if denominator != 0 else numerator

    print(f"Gradient Checking Difference: {difference:.10f}")
    

def visualize_weights(W, layer_idx=1, layer_size=25, img_shape=(28, 28)):
    plt.figure(figsize=(10, 5))
    for i in range(layer_size):
        plt.subplot(5, 5, i+1)
        # Reshape the i-th row of W[1] into 28x28
        weight_img = W[layer_idx][i, :].reshape(img_shape)
        plt.imshow(weight_img, cmap='gray')
        plt.title(f'Neuron {i+1}')
        plt.axis('off')
    plt.suptitle('Weights of First Hidden Layer (W[1])')
    plt.tight_layout()
    plt.show()


def visualize_activations(W, b, x_sample, layers, sample_idx=0):
    # Forward propagate a single sample
    a = {0: x_sample[:, sample_idx].reshape(-1, 1)}  # Shape: (784, 1)
    z = {}
    nl = len(layers)
        
    for i in range(1, nl-1):
        z[i] = np.matmul(W[i], a[i-1]) + b[i]
        a[i] = activation(z[i])
        
    z[nl-1] = np.matmul(W[nl-1], a[nl-2]) + b[nl-1]
    a[nl-1] = final_activation(z[nl-1])
    
    # Plot activations for hidden layers
    plt.figure(figsize=(12, 5))
    plt.tight_layout()
    plt.show()

start = time.time()



    
    #Tensorflow importing mnist data set   
    # Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
    
# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
    
# Reshape images to vectors (784,) and transpose for your neural network
x_train = x_train.reshape(-1, 784).T  # Shape: (784, 60000)
x_test = x_test.reshape(-1, 784).T    # Shape: (784, 10000)
    
samples_train = 6
samples_test = 1
    
# Subsample to 5000 train and 1000 test
x = x_train[:, :samples_train]  # Shape: (784, 5000)
y_train = y_train[:samples_train]      # Shape: (5000,)
x_test = x_test[:, :samples_test]    # Shape: (784, 1000)
y_test_sample = y_test[:samples_test]       # Shape: (1000,)
       
        
#variables

layers = [785, 10, 10, 1]
epoch = 8000
alpha = 0.1
lam = 10 
        
print(f"hidden layers: {layers}")
    
nl = len(layers)
    
y = one_hot(y_train,layers,nl).T
y_test  = one_hot(y_test_sample, layers, nl).T
    
a = {}
z = {}
             
m = x.shape[1]        

print(f"Lambda: {lam}") 
        
print(epoch, alpha)
#update weights and biases
W,b = init(nl,layers,x)
    
    
print("Gradient checking...........")
grad_check(x, y, W, b, layers, lam) #Gradient checking result
    
# Initialize lists to store 
cost_ = []
test_costs = []
train_accuracies = []
test_accuracies = []
    
a[0] = x
    
for k in range(0, epoch):
         #forward propagation
         z,a =  forward_prop(W, a, b, z)
         
         #backward propagation
         dW,db = backward_prop(a, y, W, b, z, layers,m)
         # gradient descent optimizer
         for j in range(1, nl):
           W[j] = W[j] - alpha * ( dW[j] + (lam/m) * W[j] )
           b[j] = b[j] - alpha * db[j]
         
         cost = cost_func(y, a[len(layers)-1], W, lam)
         cost_.append(cost)
         
         test_a = {0: x_test}
         test_z, test_a = forward_prop(W, test_a, b, {})
         test_cost = cost_func(y_test, test_a[len(layers)-1], W, lam)
         test_costs.append(test_cost)
         
         if k % 500 == 0: print(f"Epoch {k}, Train Cost: {cost:.4f}, Test Cost: {test_cost:.4f}")
         
         # --- TRAINING ACCURACY ---
         # Use the last activation from training
         train_preds = np.argmax(a[nl-1], axis=0)
         train_accuracy = np.mean(train_preds == y_train)  * 100
         train_accuracies.append(train_accuracy)
    
         test_preds = predict(W, b, x_test, layers)
         test_accuracy = np.mean(test_preds == y_test_sample)  * 100
         test_accuracies.append(test_accuracy)
         
y_hat = a[len(layers)-1]
    #num = compute_numerical_gradient(cost_func(y,y_hat,W), W[1])
    #print(num)
    
    # --- Plotting costs ---
    
    # Plot training and test cost
plt.figure(figsize=(18, 5))
    
    # Plot cost
plt.subplot(1, 2, 1)
plt.plot(cost_, label='Training Cost', color='blue')
plt.plot(test_costs, label='Test Cost', color='red')
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Training and Test Cost over Epochs")
plt.legend()
plt.grid(True)
    
    # Final accuracy report
print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    
    # --- Visualize Hidden Layers ---
visualize_weights(W, layer_idx=1, layer_size=25, img_shape=(28, 28))
visualize_activations(W, b, x_test, layers, sample_idx=0)
    
end = time.time()
    
print(f"\nTime Taken to run: {(end-start)/60} minutes")
