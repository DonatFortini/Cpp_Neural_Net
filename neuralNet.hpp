#include "matrix.hpp"
#include <math.h>

#define e 2.71828182846

class neuralNet
{
private:
    int inputSize;
    int outputSize;
    int hiddenSize;
    Matrix bias_hidden;
    Matrix bias_output;
    Matrix weights_input_hidden;
    Matrix weights_hidden_output;

public:
    neuralNet(int inputSize, int outputSize, int hiddenSize);
    ~neuralNet();
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    Matrix forward(Matrix inputs);
    void backward(Matrix inputs, Matrix targets, double learning_rate);
    void train(Matrix inputs, Matrix targets, double learning_rate,int epochs);
};

neuralNet::neuralNet(int inputSize, int outputSize, int hiddenSize)
{
    this->inputSize = inputSize;
    this->outputSize = outputSize;
    this->hiddenSize = hiddenSize;
    this->weights_input_hidden = randArray(this->inputSize, this->hiddenSize);
    this->bias_hidden = zeros(1, this->hiddenSize);
    this->weights_hidden_output = randArray(this->hiddenSize, this->outputSize);
    this->bias_output = zeros(1, this->outputSize);
}

neuralNet::~neuralNet()
{
}


