#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>

class Matrix
{
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}

    Matrix() : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}
    
    std::vector<std::vector<double>> getData()
    {
        return data;
    }

    size_t getRows() const
    {
        return rows;
    }

    size_t getCols() const
    {
        return cols;
    }

    double &operator()(size_t row, size_t col)
    {
        return data[row][col];
    }

    double operator()(size_t row, size_t col) const
    {
        return data[row][col];
    }
    void fill(std::string arg);
    Matrix dot(const Matrix &other) const;
    void print();
};

Matrix zeros(size_t rows, size_t cols);
Matrix randArray(size_t rows, size_t cols);