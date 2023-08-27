#include "neuralNet.hpp"

double NeuralNetwork::sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double NeuralNetwork::sigmoid_derivative(double x)
{
    return x * (1 - x);
}

Matrix NeuralNetwork::sigmoid(const Matrix &m)
{
    Matrix result = m;
    for (size_t i = 0; i < m.getRows(); ++i)
    {
        for (size_t j = 0; j < m.getCols(); ++j)
        {
            result(i, j) = sigmoid(m(i, j));
        }
    }
    return result;
}

Matrix NeuralNetwork::sigmoid_derivative(const Matrix &m)
{
    Matrix result = m;
    for (size_t i = 0; i < m.getRows(); ++i)
    {
        for (size_t j = 0; j < m.getCols(); ++j)
        {
            result(i, j) = sigmoid_derivative(m(i, j));
        }
    }
    return result;
}

Matrix NeuralNetwork::forward(const Matrix &inputs)
{
    Matrix hidden_output = sigmoid(dot(inputs, weights_input_hidden) + bias_hidden);
    Matrix final_output = sigmoid(dot(hidden_output, weights_hidden_output) + bias_output);
    return final_output;
}

void NeuralNetwork::backward(Matrix &inputs, Matrix &targets, double learning_rate)
{

    Matrix error = targets - forward(inputs);
    Matrix output_delta = error * sigmoid_derivative(forward(inputs));
    Matrix hidden_error = dot(output_delta, weights_hidden_output.transpose());
    Matrix hidden_delta = sigmoid_derivative(sigmoid(dot(inputs, weights_input_hidden) + bias_hidden)) * hidden_error;
    weights_hidden_output = weights_hidden_output + dot(forward(inputs).transpose(), output_delta) * learning_rate;
    bias_output = bias_output + sum(output_delta) * learning_rate;
    weights_input_hidden = weights_input_hidden + dot(inputs.transpose(), hidden_delta) * learning_rate;
    bias_hidden = bias_hidden + sum(hidden_delta) * learning_rate;
}

void NeuralNetwork::train(Matrix inputs, Matrix targets, double learning_rate, int epochs)
{
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (double i = 0; i < inputs.getCols(); ++i)
        {
            Matrix input_data = inputs[i];
            Matrix target_data = targets[i];
            Matrix output = forward(input_data);
            backward(input_data, target_data, learning_rate);
            double loss = mean(square((target_data - output)));
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Sample " << i + 1 << "/" << inputs.getCols() << ", Loss: " << loss << std::endl;
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

    NeuralNetwork net(2, 4, 1);
    net.train(inputs, targets, 0.1, 10000);

    for (double i = 0.0; i < inputs.getRows(); i++)
    {
        Matrix input_data = inputs[i];
        Matrix target_data = targets[i];
        Matrix predicted_output = net.forward(input_data);
        std::cout << "Input: ";
        input_data.print();
        std::cout << ", Target: ";
        target_data.print();
        std::cout << ", Predicted: ";
        predicted_output.print();
        std::cout << std::endl;
    }

    return 0;
}
