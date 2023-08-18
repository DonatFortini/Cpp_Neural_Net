import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.bias_output = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, inputs):
        # Calculate hidden layer output
        self.hidden_output = self.sigmoid(np.dot(inputs, self.weights_input_hidden) + self.bias_hidden)
        
        # Calculate final output
        self.final_output = self.sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output)
        
        return self.final_output
    
    def backward(self, inputs, targets, learning_rate):
        # Calculate the error
        error = targets - self.final_output
        
        # Calculate the output layer gradient
        output_delta = error * self.sigmoid_derivative(self.final_output)
        
        # Calculate the hidden layer error
        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        
        # Calculate the hidden layer gradient
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases using gradient descent
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += inputs.reshape(-1, 1).dot(hidden_delta.reshape(1, -1)) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    
    def train(self, inputs, targets, learning_rate, epochs):
        for epoch in range(epochs):
            for i in range(len(inputs)):
                input_data = inputs[i]
                target_data = targets[i]
                
                # Forward pass
                output = self.forward(input_data)
                
                # Backward pass and update weights
                self.backward(input_data, target_data, learning_rate)
                
                # Calculate and print the loss (MSE) for monitoring
                loss = np.mean(np.square(target_data - output))
                print(f"Epoch {epoch+1}/{epochs}, Sample {i+1}/{len(inputs)}, Loss: {loss:.6f}")

# Example usage
if __name__ == "__main__":
    # Sample dataset (XOR problem)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = np.array([[0], [1], [1], [0]])
    
    # Create a neural network
    input_size = 2
    hidden_size = 4
    output_size = 1
    learning_rate = 0.1
    epochs = 10000
    
    neural_net = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Train the neural network
    neural_net.train(inputs, targets, learning_rate, epochs)
    
    # Test the trained network
    for i in range(len(inputs)):
        input_data = inputs[i]
        target_data = targets[i]
        predicted_output = neural_net.forward(input_data)
        print(f"Input: {input_data}, Target: {target_data}, Predicted: {predicted_output}")
