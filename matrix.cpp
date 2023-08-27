#include "matrix.hpp"

bool Matrix::isSquare()
{
    return cols == rows;
}

Matrix Matrix::operator-(Matrix other) const
{
    if (!(((cols == rows) && other.isSquare()) || (cols == other.cols && rows == other.rows)))
        throw std::invalid_argument("Matrix dimensions do not match.");

    Matrix result(rows,cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result(i, j) = data[i][j] - other.data[i][j];
        }
    }

    return result;
}

Matrix Matrix::operator+(Matrix other) const
{
    if (!(((cols == rows) && other.isSquare()) || (cols == other.cols && rows == other.rows)))
        throw std::invalid_argument("Matrix dimensions do not match.");

    Matrix result(rows,cols);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result(i, j) = data[i][j] + other.data[i][j];
        }
    }

    return result;
}

Matrix Matrix::operator*(Matrix other) const
{
    if (other.rows != cols)
        throw std::invalid_argument("Matrix dimensions do not match for dot product.");

    Matrix result(rows, other.cols);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < other.cols; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < cols; ++k)
            {
                sum += data[i][k] * other(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

Matrix Matrix::operator*(double other) const
{
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result(i, j) = data[i][j] * other;
        }
    }

    return result;
}

Matrix Matrix::operator=(double other) const
{
    return Matrix();
}

double Matrix::operator-(double other) const
{
    if (!(cols == rows && cols == 1))
        throw std::invalid_argument("Matrix dimensions do not match.");
    return data[0][0] - other;
}

double Matrix::operator+(double other) const
{
    if (!(cols == rows && cols == 1))
        throw std::invalid_argument("Matrix dimensions do not match.");
    return data[0][0] + other;
}

Matrix Matrix::operator[](double index) const
{
    std::vector<double> slice = data[index];
    Matrix x(1, (int)slice.capacity());
    for (int i = 0; i < (int)slice.capacity(); i++)
    {
        x(0, i) = slice.at(i);
    }

    return x;
}

double Matrix::operator[](int index) const
{
    return data[0][index];
}

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

Matrix dot(const Matrix &origin, const Matrix &other)
{
    int rows = origin.getRows();
    int cols = other.getCols();

    if (origin.getCols() != other.getRows())
        throw std::invalid_argument("Matrix dimensions do not match for dot product.");

    Matrix result(rows, cols);

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < origin.getCols(); ++k)
            {
                sum += origin(i, k) * other(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

double dot(const double &origin, const double &other)
{
    return origin * other;
}

double sum(Matrix &A)
{
    double sum = 0.0;
    int count = 0;
    for (const auto &iter : A.getData())
    {
        for (const double &i : iter)
        {
            sum += i;
            ++count;
        }
    }
    return sum;
}

Matrix square(Matrix A)
{
    return dot(A, A);
}

double mean(Matrix A)
{
    return sum(A) / (A.getCols() * A.getRows());
}

Matrix zeros(int rows, int cols)
{
    Matrix zero(rows, cols);
    zero.fill("zero");
    return zero;
}

Matrix randArray(int rows, int cols)
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

Matrix Matrix::transpose()
{
    Matrix result(cols, rows);
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            result(j, i) = data[i][j];
        }
    }

    return result;
}

/*
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

    Matrix inputs(4, 2);
    inputs(0, 0) = 0.0;
    inputs(0, 1) = 0.0;
    inputs(1, 0) = 0.0;
    inputs(1, 1) = 1.0;
    inputs(2, 0) = 1.0;
    inputs(2, 1) = 0.0;
    inputs(3, 0) = 1.0;
    inputs(3, 1) = 1.0;

    Matrix z(2, 2);
    z(0, 0) = 1.0;
    z(0, 1) = 2.0;
    z(1, 0) = 3.0;
    z(1, 1) = 4.0;

    Matrix C = dot(A, B);
    Matrix D = zeros(4, 2);
    Matrix x = randArray(3, 3);
    C.print();
    D.print();
    x.print();
    inputs.print();
    std::cout << mean(z) << std::endl;
    std::cout << std::endl;
    z.print();
    z = z.transpose();
    double ezae = 2.0;
    z = z * ezae;
    z.print();

    inputs[1.0].print();

    return 0;
}
*/