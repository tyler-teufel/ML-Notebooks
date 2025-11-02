# Recurrent Neural Networks (RNNs)

## RNN Basics

### What is an RNN?

Recurrent Neural Networks (RNNs) are a class of neural networks specifically designed to process sequential data. Unlike feedforward neural networks, which assume all inputs are independent, RNNs leverage the sequential nature of data by maintaining a "memory" of previous inputs using hidden states.

**Key Idea:** The output at any time step is influenced not just by the input at that time but also the information from previous steps.

- produces predictive results in sequential data that other algorithms cannot!
- RNNs have an "internal state" that is updated as a sequence is processed.

### Why use RNNs?

RNNs are ideal for tasks involving:

- Time series analysis (stock prices, weather forecasting)
- Natural language Processing (NLP)
- Sequential Decision-making (Reinforcement learning in games)

### Architecture of RNNs

- **Input Layer:** processes one element of the sequence at each time step.
- **Hidden Layer:** The "recurrent" component that updates its state each step by combining the input and the previous hidden state.
- **Output layer:** Produces predictions or representations at each step, depending on the task.

Mathematically:

$$
h_t = tanh(W_h ​⋅ h_{t-1} + W_x ​⋅ x_t + b)
$$

$$
y_t = softmax(W_y ​⋅ h_t + c)
$$

Where:

- $h_t

We can process a sequence of vectors $x$ by applying a **Recurrence formula** at every time step:

$$
h_t = fw(h_{t-1},x_t)
$$

where $h_t$ represents the new state, $fw$ represents some function with parameters $w$, $h_{t-1}$ represents the old state, and $x_t$ represents the input vector at some step.
