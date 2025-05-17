import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()                    # Load data from Keras into train/test split

xTrain = xTrain.reshape(-1, 784)                                        # Convert the 2dpxel array into a one dimensional 
xTest = xTest.reshape(-1, 784)                                          # value so we can input into generic neural network

xTrain = xTrain.astype("float32") / 255.0                               # Normalize both to increase accuracy 
xTest = xTest.astype("float32") / 255.0 

yTrainDig = to_categorical(yTrain, 10)                                  # convert the outputs into categoricals
yTestDig = to_categorical(yTest, 10)


# Function Design: 
def relu(x):                                                            # simple RELU
    return np.maximum(0, x)

def reluDer(x):                                                         # RELU Derivative for gradient descent
    return (x > 0).astype(float)

def softmax(x):                                                         # Softmax to convert raw outputs into probabilities of each digit
    ex = np.exp(x - np.max(x, axis = 1, keepdims = True))
    return ex / np.sum(ex, axis = 1, keepdims = True)

def loss(pred, true):                                                   # Loss to calculate how far each output probability is from the actual probability 
    return -np.sum(true * np.log(pred + 1e-8)) / pred.shape[0]

def accuracy(pred, true):                                               # Accuracy for model post processing. 
    return np.mean(np.argmax(pred, axis = 1) == np.argmax(true, axis = 1))    

def forward(x, W1, b1, W2, b2, W3, b3):                                 # 3 layers total for the forward pass through the model 
    z1 = x @ W1 + b1 
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    a3 = softmax(z3)
    return z1, a1, z2, a2, z3, a3

def backward(x, y, z1, a1, z2, a2, z3, a3, W2 ,W3):                     # BackPropagation (ouput->input) to adjust our wieghts and biases
    bSize = x.shape[0]
    
    g3 = (a3 - y) / bSize                                               # Output layer Gradient  
    dW3 = a2.T @ g3                                                     # transpose the output of second layer so that we can get the access to the weights 
    db3 = np.sum(g3, axis = 0, keepdims = True)                         # since shape of a2 is (num inputs, num outputs (shich is the same number as the numebr of weights))
    
    da2 = g3 @ W3.T                                                     # gradient is  now travelling back to next layer 
    dz2 = da2 * reluDer(z2)                                             # da2 is the gchange in total loss if there was a slight change to the a2 values
    dW2 = a1.T @ dz2                                                    # gain access to the weights 
    db2 = np.sum(dz2, axis = 0, keepdims = True)
    
    da1 = dz2 @ W2.T                                                    # gradient is  now travelling back to next layer 
    dz1 = da1 * reluDer(z1)                                             # da1 is the gchange in total loss if there was a slight change to the a1 values
    dW1 = x.T @ dz1                                                     # gain access to the weights 
    db1 = np.sum(dz1, axis = 0, keepdims = True)
    
    return dW1, db1, dW2, db2, dW3, db3

def update(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, lr):   # updates the weights and biases
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2
    W3 -= lr * dW3
    b3 -= lr * db3
    
    return W1, b1, W2, b2, W3, b3

# Initialization: 
inputsSize = 784                                                        # 784 pixels inputted
hiddenSize1 = 256                                                       # arbitrarily chosen 
hiddenSize2 = 128                                                       # arbitrarily chosen 
outputSize = 10                                                         # 0-9 digits
epochs = 10                                                             # go through the training data 10 times
lr = 0.1                                                                # tweak gradient desecnt change to be 0.1 amount
batchSize = 100                                                         # Chosen to split training into groups of 100 for gradient descent
W1 = W1 = np.random.randn(inputsSize, hiddenSize1) * 0.01
b1 = np.zeros((1, hiddenSize1))

W2 = np.random.randn(hiddenSize1, hiddenSize2) * 0.01
b2 = np.zeros((1, hiddenSize2))

W3 = np.random.randn(hiddenSize2, outputSize) * 0.01
b3 = np.zeros((1, outputSize))

# Training Loop: 
for epoch in range(epochs): 
    totalLoss = 0
    for i in range(0, xTrain.shape[0], batchSize): 
        xBatch = xTrain[i:i+batchSize]
        yBatch = yTrainDig[i:i+batchSize]
        
        z1, a1, z2, a2, z3, a3 = forward(xBatch, W1, b1, W2, b2, W3, b3)
        
        lossVal = loss(a3, yBatch)
        totalLoss += lossVal
        
        dW1, db1, dW2, db2, dW3, db3 = backward(xBatch, yBatch, z1, a1, z2, a2, z3, a3, W2 ,W3)
        
        W1, b1, W2, b2, W3, b3 = update(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, lr)
        
    print(f"Epoch {epoch+1}, Loss: {totalLoss:.4f}")
    
    
# Evaluation: 
z1Test = xTest @ W1 + b1
a1Test = relu(z1Test)
z2Test = a1Test @ W2 + b2
a2Test = relu(z2Test)
z3Test = a2Test @ W3 + b3
a3Test = softmax(z3Test)

accuracyVal = accuracy(a3Test, yTestDig)
print(f"Test Accuracy: {accuracyVal:.4f}")