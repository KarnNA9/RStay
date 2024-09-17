# Week 6: Long short-term memory (LSTM) and gated recurrent units (GRU)


# Long Short-Term Memory (LSTM)
Long Short-Term Memory (LSTM) networks are a type of recurrent neural network (RNN) introduced by Hochreiter and Schmidhuber in 1997. They're specifically designed to avoid the long-term dependency problem, a challenge with standard RNNs, which have trouble learning to connect information that is spaced far apart in time sequences.

LSTM networks include a 'memory cell' that can maintain information in memory for long periods. As the network processes sequences of data, it can selectively read from, write to, or forget its memory cell using structures called gates. These gates are essentially neural networks themselves, which learn to control the memory cell's behavior.

Each LSTM unit has three such gates:

Forget Gate: Decides what information should be thrown away or kept.
Input Gate: Updates the cell state with new information.
Output Gate: Determines what the next hidden state should be.
The critical advantage of LSTMs is their ability to remember from long-term sequences (window sizes), which is extremely useful in many NLP tasks.

# Gated Recurrent Units (GRU)
Gated Recurrent Units (GRU) are another variant of recurrent neural networks, introduced by Cho, et al. in 2014. They're similar to LSTMs with a forget and an input gate, but they lack an output gate. Also, the memory cell and hidden state are merged into a single context unit, resulting in a simpler and more streamlined architecture.

In particular, GRUs have two gates:

Reset Gate: Determines how much of the previous state to forget.
Update Gate: Controls how much of the state information to update with new data.
Due to their reduced complexity, GRUs are often faster to train than LSTMs. However, they may not perform as well on tasks requiring sophisticated memory usage as they lack an output gate.

LSTM and GRU in Practice
LSTM and GRU units are especially useful for sequence prediction problems because they can use their memory to process sequences of arbitrary length, instead of assuming a pre-specified window as input, like in traditional neural networks. They're widely used in tasks such as language modeling, machine translation, speech recognition, and much more.

For example, in the task of text generation, an LSTM or GRU network can learn the probability of a word given its preceding words, thus generating a sequence that follows the learned language patterns. The memory cells in these networks allow them to maintain a form of 'context' by remembering the past words, which influences the upcoming word predictions.

# Readings

[Long Sort-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)

[Tutorial on GRU](https://d2l.ai/chapter_recurrent-modern/gru.html)

[Using LSTM and GRU neural network methods for traffic flow prediction](https://www.researchgate.net/profile/Li-Li-86/publication/312402649_Using_LSTM_and_GRU_neural_network_methods_for_traffic_flow_prediction/links/5c20d38d299bf12be3971696/Using-LSTM-and-GRU-neural-network-methods-for-traffic-flow-prediction.pdf)

Additional article

[Peak Electrical Energy Consumption Prediction by ARIMA, LSTM, GRU, ARIMA-LSTM and ARIMA-GRU Approaches ](https://www.mdpi.com/1996-1073/16/12/4739)

# Code examples

# Example 1

In this example, we explore the Long Short-Term Memory (LSTM) neural network, which is a variant of recurrent neural networks (RNNs). LSTMs are particularly useful for sequential data tasks, such as time series forecasting or natural language processing, where there are long-term dependencies between data points.

Data Generation:
We begin by generating synthetic sequential data for our LSTM model. The data consists of two sequences, each containing ten consecutive numbers. The target is the third number in each sequence. We convert this data into PyTorch tensors, which is the format required for processing with PyTorch.

LSTM Model Architecture:
The LSTM model is defined using the LSTMModel class. Inside the class, we set up the LSTM layer, which takes a sequence of one-dimensional inputs and processes them in a time-ordered manner. We also include a fully connected layer to predict the next number in the sequence. The model architecture captures the patterns and dependencies present in the input sequences.

Forward Pass:
The forward method within the LSTMModel class describes how data flows through the model. During the forward pass, the LSTM processes the input sequences, capturing important information over time. The hidden state and cell state are initialized at the beginning of each forward pass and updated as the LSTM processes the data.

Loss Function and Optimizer:
For training, we define the Mean Squared Error (MSE) loss function to quantify the difference between predicted and actual values. To optimize the model's parameters during training, we use the Adam optimizer, which adapts the learning rate based on the gradients of the model's parameters.

Training Loop:
We run a loop for a specified number of epochs to train the LSTM model. In each epoch, we forward propagate the input sequences through the model, compute the loss, perform backpropagation to calculate gradients, and update the model's parameters using the optimizer. The goal is to minimize the loss and improve the model's prediction accuracy.

Testing the Trained Model:
After training, we evaluate the LSTM model's performance on a test sequence, [10, 11]. The trained LSTM predicts the next number in the sequence based on the learned patterns from the training data.

# Example 2

In this example, we explore the Gated Recurrent Unit (GRU), which is a variant of recurrent neural networks (RNNs). Like LSTMs, GRUs are particularly useful for sequential data tasks, but they have fewer parameters and are often computationally faster.

Data Generation:
We start by generating synthetic sequential data, which consists of two sequences, each containing ten consecutive numbers. The target is the third number in each sequence. We convert this data into PyTorch tensors, preparing it for processing with PyTorch.

GRU Model Architecture:
The GRU model is defined using the GRUModel class. Inside this class, we set up the GRU layer, which takes a sequence of one-dimensional inputs and processes them in a time-ordered manner. Additionally, we include a fully connected layer to predict the next number in the sequence. The model architecture captures the patterns and dependencies present in the input sequences.

Forward Pass:
The forward method within the GRUModel class describes how data flows through the model. During the forward pass, the GRU processes the input sequences, capturing important information over time. The hidden state is initialized at the beginning of each forward pass and updated as the GRU processes the data.

Loss Function and Optimizer:
For training, we define the Mean Squared Error (MSE) loss function to measure the difference between predicted and actual values. To optimize the model's parameters during training, we use the Adam optimizer, which adapts the learning rate based on the gradients of the model's parameters.

Training Loop:
We run a loop for a specified number of epochs to train the GRU model. In each epoch, we forward propagate the input sequences through the model, compute the loss, perform backpropagation to calculate gradients, and update the model's parameters using the optimizer. The goal is to minimize the loss and improve the model's prediction accuracy.

Testing the Trained Model:
After training, we evaluate the GRU model's performance on a test sequence, [10, 11]. The trained GRU predicts the next number in the sequence based on the learned patterns from the training data.