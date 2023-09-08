#include "matrix.hpp"
#include <math.h>

class NeuralNetwork
{
private:
    Matrix weights_input_hidden;
    Matrix bias_hidden;
    Matrix weights_hidden_output;
    Matrix bias_output;
    Matrix hidden_output;
    Matrix final_output;
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size)
        : weights_input_hidden(input_size, hidden_size), bias_hidden(1, hidden_size),
          weights_hidden_output(hidden_size, output_size), bias_output(1, output_size)
    {
        weights_input_hidden.fill("rand");
        bias_hidden.fill("zero");
        weights_hidden_output.fill("rand");
        bias_output.fill("zero");
    }
    Matrix getFinal(void);
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    Matrix sigmoid(const Matrix &m);
    Matrix sigmoid_derivative(const Matrix &m);
    void forward(const Matrix &inputs);
    void backward(Matrix &inputs, Matrix &targets, double learning_rate);
    void train(Matrix inputs, Matrix targets, double learning_rate, int epochs);
};
