#include "matrix.hpp"

bool Matrix::isSquare(void)
{
    return cols == rows;
}

double &Matrix::operator()(int row, int col)
{
    return data[row][col];
}

double Matrix::operator()(int row, int col) const
{
    return data[row][col];
}

Matrix Matrix::operator-(Matrix other) const
{
    if (!(((cols == rows) && other.isSquare()) || (cols == other.cols && rows == other.rows)))
        throw std::invalid_argument("Matrix dimensions do not match.");

    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i, j) = data[i][j] - other.data[i][j];
    return result;
}

Matrix Matrix::operator+(Matrix other) const
{
    if (!(((cols == rows) && other.isSquare()) || (cols == other.cols && rows == other.rows)))
        throw std::invalid_argument("Matrix dimensions do not match.");

    Matrix result(rows, cols);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(i, j) = data[i][j] + other.data[i][j];
    return result;
}

Matrix Matrix::operator*(Matrix other) const
{
    if (other.rows != cols)
        throw std::invalid_argument("Matrix dimensions do not match for dot product.");

    Matrix result(rows, other.cols);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < other.cols; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < cols; ++k)
                sum += data[i][k] * other(k, j);
            result(i, j) = sum;
        }
    return result;
}

Matrix Matrix::operator*(double other) const
{
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result(i, j) = data[i][j] * other;
    return result;
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
        x(0, i) = slice.at(i);
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
        for (double &i : iter)
            i = (arg == "zero") ? 0.0 : static_cast<double>(std::rand()) / RAND_MAX;
}

Matrix dot(const Matrix &origin, const Matrix &other)
{
    int rows = origin.getRows();
    int cols = other.getCols();

    if (origin.getCols() != other.getRows())
        throw std::invalid_argument("Matrix dimensions do not match for dot product.");

    Matrix result(rows, cols);

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
        {
            double sum = 0.0;
            for (int k = 0; k < origin.getCols(); ++k)
                sum += origin(i, k) * other(k, j);
            result(i, j) = sum;
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
        for (const double &i : iter)
        {
            sum += i;
            ++count;
        }
    return sum;
}

Matrix sum(Matrix A, int axis, bool kd)
{
    if (kd)
    {
        if (axis != 0 && axis != 1)
            throw std::invalid_argument("axis unavailable!!");
        else
        {
            if (axis == 0)
            {
                Matrix x(1, A.getCols());
                for (int i = 0; i < A.getCols(); i++)
                {
                    double s = 0;
                    for (int j = 0; j < A.getRows(); j++)
                    {
                        s += A.getData()[j][i];
                    }

                    x(0, i) = s;
                }

                return x;
            }
            else
            {
                Matrix x(A.getRows(), 1);
                for (int i = 0; i < A.getRows(); i++)
                {
                    double s = 0;
                    for (int j = 0; j < A.getCols(); j++)
                    {
                        s += A.getData()[i][j];
                    }

                    x(i, 0) = s;
                }
                return x;
            }
        }
    }
    else
    {
        Matrix x(1, 1);
        x(0, 0) = sum(A);
        return x;
    }
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

void Matrix::print(void)
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

Matrix Matrix::transpose(void) const
{
    Matrix result(cols, rows);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result(j, i) = data[i][j];
    return result;
}

Matrix Matrix::linear(Matrix same) const
{
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            result(i, j) = data[i][j] * same(i, j);
    return result;
}

Matrix Matrix::reshape(int Newrows, int Newcols) const
{
    if (Newcols == -1 && Newrows == -1)
        throw std::invalid_argument("wrong dimensions.");

    Newrows = (Newrows == -1) ? (cols * rows) / Newcols : Newrows;
    Newcols = (Newcols == -1) ? (cols * rows) / Newrows : Newcols;

    if (cols * rows != Newcols * Newrows)
        throw std::invalid_argument("Matrix dimensions do not match for reshape.");

    double line[Newcols * Newrows];
    Matrix result(Newrows, Newcols);
    int x = 0;

    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
        {
            line[x] = data[i][j];
            ++x;
        }

    x = 0;

    for (int i = 0; i < Newrows; ++i)
        for (int j = 0; j < Newcols; ++j)
        {
            result(i, j) = line[x];
            ++x;
        }

    return result;
}

/* test
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

    B.reshape(1, 6).print();
    B.reshape(1, -1).print();

    Matrix s(3, 2);
    s(0, 0) = 0.0;
    s(0, 1) = 1.0;
    s(1, 0) = 2.0;
    s(1, 1) = 3.0;
    s(2, 0) = 4.0;
    s(2, 1) = 5.0;

    s = sum(s, 1, false);
    s.print();

    return 0;
}
*/