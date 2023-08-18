#include "matrix.hpp"

void Matrix::fill(std::string arg)
{
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (auto &iter : this->data)
    {
        for (double &i : iter)
        {
            i = (arg == "zero") ? 0.0 : static_cast<double>(std::rand()) / RAND_MAX;
        }
    }
}

Matrix Matrix::dot(const Matrix &other) const
{
    if (cols != other.rows)
        throw std::invalid_argument("Matrix dimensions do not match for dot product.");
    Matrix result(rows, other.cols);

    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < other.cols; ++j)
        {
            double sum = 0.0;
            for (size_t k = 0; k < cols; ++k)
            {
                sum += data[i][k] * other(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

Matrix zeros(size_t rows, size_t cols)
{
    Matrix zero(rows, cols);
    zero.fill("zero");
    return zero;
}

Matrix randArray(size_t rows, size_t cols)
{
    Matrix randArr(rows, cols);
    randArr.fill("rand");
    return randArr;
}

void Matrix::print()
{
    for (const auto &iter : this->data)
    {
        for (const double &i : iter)
        {
            std::cout << i << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main()
{
    Matrix A(2, 3);
    A(0, 0) = 1.0;
    A(0, 1) = 2.0;
    A(0, 2) = 3.0;
    A(1, 0) = 4.0;
    A(1, 1) = 5.0;
    A(1, 2) = 6.0;

    Matrix B(3, 2);
    B(0, 0) = 7.0;
    B(0, 1) = 8.0;
    B(1, 0) = 9.0;
    B(1, 1) = 10.0;
    B(2, 0) = 11.0;
    B(2, 1) = 12.0;

    Matrix inputs(4,2);
    inputs(0, 0) = 0.0;
    inputs(0, 1) = 0.0;
    inputs(1, 0) = 0.0;
    inputs(1, 1) = 1.0;
    inputs(2, 0) = 1.0;
    inputs(2, 1) = 0.0;
    inputs(3, 0) = 1.0;
    inputs(3, 1) = 1.0;

    Matrix C = A.dot(B);
    Matrix D = zeros(4, 2);
    Matrix x = randArray(3, 3);
    C.print();
    D.print();
    x.print();

    inputs.print();
    return 0;
}
