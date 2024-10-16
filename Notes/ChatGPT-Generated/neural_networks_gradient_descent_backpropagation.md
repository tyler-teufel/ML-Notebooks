
# Neural Networks, Gradient Descent, and Backpropagation

## 1. What is a Neural Network?

A neural network is a computational model inspired by the human brain, consisting of layers of nodes (neurons) that can learn from data by adjusting their weights.

### Key Components:
- **Input Layer**: Takes in the features of the dataset (e.g., in an image recognition model, each pixel value is an input).
- **Hidden Layers**: These are the layers between the input and output, where the actual computation happens. They process and transform the inputs.
- **Output Layer**: Provides the final prediction, such as classifying whether an image shows a cat or a dog.

Each neuron in one layer is connected to neurons in the next layer by **weights**, which determine how much influence one neuron has on another.

## 2. Activation Functions and the Sigmoid Function

Each neuron in the hidden and output layers applies an **activation function** to its input to produce its output. The **sigmoid function** is a commonly used activation function, especially for binary classification problems.

### The Sigmoid Function:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Here, \( z \) is the weighted sum of the inputs to the neuron, plus a bias term:
$$z = \sum_{i=1}^{n} w_i x_i + b$$

The sigmoid function squashes the output into a range between 0 and 1, making it useful for interpreting the result as a probability.

## 3. How Neural Networks Learn: Forward Pass and Loss Function

### Forward Pass:
1. In the **forward pass**, the input data is passed through the network, layer by layer.
2. Each neuron computes a **weighted sum** of its inputs, applies the **activation function** (such as the sigmoid), and sends the result to the next layer.

### Loss Function:
Once the input passes through the entire network and reaches the output layer, the network computes a **loss** (also called the cost) that measures the difference between the network’s prediction and the actual target value.

For binary classification, a common loss function is the **binary cross-entropy loss**:
$$J(\theta) = - \frac{1}{m} \sum_{i=1}^{m} [y_i \log(h_\theta(x_i)) + (1 - y_i) \log(1 - h_\theta(x_i))]$$
where $h_\theta(x)$ is the predicted output, $y$ is the actual label, and $m$ is the number of training examples.

## 4. Gradient Descent in Neural Networks

Once we calculate the loss, we want to minimize it by adjusting the weights in the network. This is where **gradient descent** comes in.

Gradient descent updates the weights by computing the gradient of the loss function with respect to the weights. In neural networks, the loss depends on all the weights in the network, and they’re interconnected through the layers. To adjust the weights, we compute **how changing each weight affects the loss**.

## 5. Backpropagation: Connecting Gradient Descent and Neural Networks

**Backpropagation** is the algorithm that computes the gradients of the loss function with respect to each weight in the network. It uses the **chain rule** to propagate the error backward through the network and adjust the weights.

### How Backpropagation Works:

#### Step 1: Compute the Loss (Forward Pass)
First, perform a forward pass to calculate the loss by feeding the inputs through the network and obtaining the prediction.

#### Step 2: Compute the Gradient of the Loss with Respect to Output (Backward Pass)
For the output layer neuron with a sigmoid activation function:
1. Compute the **derivative of the loss** with respect to the output neuron’s output $a$:
   $$
   \frac{\partial J}{\partial a}
   $$
2. Compute the **derivative of the sigmoid function** with respect to its input $z$ (the weighted sum):
   $$
   \frac{\partial a}{\partial z} = a(1 - a)
   $$
   
#### Step 3: Chain Rule to Compute Gradients for Hidden Layers
Using the **chain rule**, the error is propagated back through the hidden layers:
$$
\frac{\partial J}{\partial w_i} = \frac{\partial J}{\partial z} \cdot \frac{\partial z}{\partial w_i}
$$

#### Step 4: Update Weights Using Gradient Descent
Weights are updated using the rule:
$$
w_i := w_i - \alpha \frac{\partial J}{\partial w_i}
$$

## 6. Tying Everything Together

- **Forward Pass**: The input is passed through the network, the weighted sums and activations are computed, and a prediction is made. The loss is then calculated.
- **Backpropagation**: The loss is used to compute the gradients of the loss with respect to each weight by propagating the error backward through the layers.
- **Gradient Descent**: The weights are updated using gradient descent, adjusting each weight based on how much it contributed to the error.

## 7. Why the Sigmoid Function?

The **sigmoid function** is useful because:
1. It outputs a value between 0 and 1, making it great for binary classification.
2. It’s differentiable, which is essential for backpropagation.
3. The derivative is simple:
   $$
   \frac{d}{dz} \sigma(z) = \sigma(z)(1 - \sigma(z))
   $$
   
## Summary

1. **Neural Networks**: Layers of neurons calculate weighted sums, apply activations, and make predictions.
2. **Gradient Descent**: Minimizes the loss by updating the network’s weights.
3. **Backpropagation**: Computes gradients to adjust weights efficiently by propagating errors backward.
4. **Sigmoid Function**: Common activation for binary classification, useful for backpropagation due to its differentiability.
