#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <string>

class Matrix
{
private:    
    int rows;
    int cols;
    std::vector<std::vector<double>> data;
public:
    Matrix(int rows, int cols) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}

    Matrix() : rows(0), cols(0) {}

    std::vector<std::vector<double>> getData(void)
    {
        return data;
    }

    int getRows(void) const
    {
        return rows;
    }

    int getCols(void) const
    {
        return cols;
    }

    bool isSquare(void);

    double &operator()(int row, int col);
    double operator()(int row, int col) const;
    Matrix operator-(Matrix other) const;
    Matrix operator+(Matrix other) const;
    Matrix operator*(Matrix other) const;
    Matrix operator*(double other) const;
    double operator-(double other) const;
    double operator+(double other) const;
    Matrix operator[](double index) const;
    double operator[](int index) const;

    void fill(std::string arg);
    void print(void);
    Matrix transpose(void) const;
    Matrix linear(Matrix same) const;
    Matrix reshape(int Newrows, int Newcols) const;
};

Matrix dot(const Matrix &origin, const Matrix &other);
double dot(const double &origin, const double &other);
double sum(Matrix &A);
Matrix sum(Matrix A, int axis, bool kd);
Matrix square(Matrix A);
double mean(Matrix A);
Matrix zeros(int rows, int cols);
Matrix randArray(int rows, int cols);