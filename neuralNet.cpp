#include "neuralNet.hpp"

double neuralNet::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double neuralNet::sigmoid_derivative(double x)
{
    return x * (1 - x);
}

Matrix neuralNet::forward(Matrix inputs)
{
    Matrix finalOutput(1, 1);
    Matrix finalOutput(1, 1);
    hidden_output = sigmoid(inputs.dot(weights_hidden_output) + bias_hidden);
    finalOutput(0, 0) = sigmoid(hidden_output.dot(weights_hidden_output) + bias_output);
    return finalOutput;
}

void neuralNet::backward(Matrix inputs, Matrix targets, double learning_rate)
{
    Matrix final_output = forward(inputs);
    error = targets - final_output;
    output_delta = error * sigmoid_derivative(final_output);
    hidden_error = output_delta.dot(weights_hidden_output.T);
    hidden_delta = hidden_error * sigmoid_derivative(hidden_output);
    weights_hidden_output += hidden_output.T.dot(output_delta) * learning_rate;
    bias_output += np.sum(output_delta, axis = 0, keepdims = True) * learning_rate;
    weights_input_hidden += inputs.reshape(-1, 1).dot(hidden_delta.reshape(1, -1)) * learning_rate;
    bias_hidden += np.sum(hidden_delta, axis = 0, keepdims = True) * learning_rate;
}

void neuralNet::train(Matrix inputs, Matrix targets, double learning_rate, int epochs)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (size_t i = 0; i < inputs.getRows(); i++)
        {
            Matrix input_data = inputs[i];
            Matrix target_data = targets[i];

            output = forward(input_data);

            backward(input_data, target_data, learning_rate);

            loss = np.mean(np.square(target_data - output));
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Sample " << i + 1 << "/" << inputs.getRows() << ", Loss: " << loss << std::endl;
        }
    }
}

int main(int argc, char const *argv[])
{
    Matrix inputs(4, 2);
    inputs(0, 0) = 0.0;
    inputs(0, 1) = 0.0;
    inputs(1, 0) = 0.0;
    inputs(1, 1) = 1.0;
    inputs(2, 0) = 1.0;
    inputs(2, 1) = 0.0;
    inputs(3, 0) = 1.0;
    inputs(3, 1) = 1.0;

    Matrix targets(4, 1);
    targets(0, 0) = 0.0;
    targets(1, 0) = 1.0;
    targets(2, 0) = 1.0;
    targets(3, 0) = 0.0;

    neuralNet net(2, 4, 1);
    net.train(inputs, targets, 0.1, 10000);

    for (size_t i = 0; i < inputs.getRows(); i++)
    {
        Matrix input_data = inputs[i];
        Matrix target_data = targets[i];
        Matrix predicted_output = net.forward(input_data);
        std::cout << "Input: " << input_data.print() << ", Target: " << target_data.print() << ", Predicted: " << predicted_output.print() << std::endl;
    }
    
    return 0;
}
