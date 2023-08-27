#include "matrix.hpp"
#include <math.h>

class NeuralNetwork
{
private:
    Matrix weights_input_hidden;
    Matrix bias_hidden;
    Matrix weights_hidden_output;
    Matrix bias_output;

public:
    NeuralNetwork(size_t input_size, size_t hidden_size, size_t output_size)
        : weights_input_hidden(input_size, hidden_size), bias_hidden(1, hidden_size),
          weights_hidden_output(hidden_size, output_size), bias_output(1, output_size)
    {
        weights_input_hidden.fill("rand");
        bias_hidden.fill("zero");
        weights_hidden_output.fill("rand");
        bias_output.fill("zero");
    }

    double sigmoid(double x);
    double sigmoid_derivative(double x);
    Matrix sigmoid(const Matrix &m);
    Matrix sigmoid_derivative(const Matrix &m);
    Matrix forward(const Matrix &inputs);
    void backward(Matrix &inputs, Matrix &targets, double learning_rate);
    void train(Matrix inputs, Matrix targets, double learning_rate, int epochs);
};
