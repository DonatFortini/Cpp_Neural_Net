#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>

class Matrix
{
private:
    std::vector<std::vector<double>> data;
    int rows;
    int cols;

public:
    Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}

    Matrix() : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}

    std::vector<std::vector<double>> getData()
    {
        return data;
    }

    int getRows() const
    {
        return rows;
    }

    int getCols() const
    {
        return cols;
    }

    bool isSquare();

    double &operator()(int row, int col)
    {
        return data[row][col];
    }

    double operator()(int row, int col) const
    {
        return data[row][col];
    }

    Matrix operator-(Matrix other) const;
    Matrix operator+(Matrix other) const;
    Matrix operator*(Matrix other) const;
    Matrix operator*(double other) const;
    Matrix operator=(double other) const;
    double operator-(double other) const;
    double operator+(double other) const;
    Matrix operator[](double index) const;
    double operator[](int index) const;

    void fill(std::string arg);
    void print();
    Matrix transpose();
    Matrix linear(Matrix same) const;
    Matrix reshape(int Newrows ,int Newcols)const;
};

Matrix dot(const Matrix &origin, const Matrix &other);
double dot(const double &origin, const double &other);
double sum(Matrix &A);
Matrix square(Matrix A);
double mean(Matrix A);
Matrix zeros(int rows, int cols);
Matrix randArray(int rows, int cols);