#include "Matrix.h"

Matrix::Matrix(int row, int column, bool randomize)
    :
    row(row),
    column(column)
{
    for (int i = 0; i < row; i++)
    {
        std::vector<double> r;
        for (int j = 0; j < column; j++)
        {
            if (randomize)
            {                                               //Ex.
                r.emplace_back(Randomize());                //r=[1, 2, 3]
            }
            else
                r.emplace_back(0.0);
        }
        values.emplace_back(r);                             //values=[1, 2, 3   ->row 1
    }                                                      //         4, 5, 6]  ->row 2
}

double Matrix::Randomize()
{
    std::random_device rd;
    std::mt19937 random(rd());
    std::uniform_real_distribution<double> Dist(0, 1);
    return Dist(random);
}

Matrix Matrix::operator*(Matrix& rhs)
{
    assert(column == rhs.row);
    Matrix c = Matrix(row, rhs.column, false);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < rhs.column; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < rhs.row; k++)
            {
                sum += values.at(i).at(k) * rhs.values.at(k).at(j);

            }
            c.SetValue(i, j, sum);
            sum = 0.0;
        }
    }
    return c;
}

Matrix Matrix::operator*(double& rhs)
{
    Matrix c = Matrix(row, column, false);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            double val = values.at(i).at(j) * rhs;
            c.SetValue(i, j, val);
        }
    }
    return c;
}

Matrix& Matrix::operator*=(double& rhs)
{
    return *this = *this * rhs;
}

//Matrix& Matrix::operator*(Matrix& rhs)
//{
//    assert(column == rhs.row);
//    for (int i = 0; i < row; i++)
//    {
//        for (int j = 0; j < rhs.column; j++)
//        {
//            double sum = 0.0;
//            for (int k = 0; k < rhs.row; k++)
//            {
//                double z = values.at(i).at(k);
//                double zz = rhs.values.at(k).at(j);
//                sum += values.at(i).at(k) * rhs.values.at(k).at(j);
//            }
//            values.at(i).at(j) = sum;
//            sum = 0.0;
//        }
//    }
//    return *this;
//}

Matrix Matrix::Hadamard(Matrix& m1, Matrix& m2)
{
    assert(m1.row == m2.row && m1.column == m2.column);
    Matrix c = Matrix(m1.row, m1.column, false);
    for (int i = 0; i < m1.row; i++)
    {
        for (int j = 0; j < m1.column; j++)
        {
            double value = m1.GetValue(i, j) * m2.GetValue(i, j);
            c.SetValue(i, j, value);
        }
    }
    return c;

}

Matrix Matrix::operator-(double value) const
{
    Matrix c = Matrix(row, column, false);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            c.SetValue(i, j, values.at(i).at(j) - value);
        }
    }
    return c;
}

Matrix Matrix::operator-(Matrix& rhs) const
{
    assert(row == rhs.row && column == rhs.column);
    Matrix c = Matrix(row, column, false);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            c.SetValue(i, j, values.at(i).at(j) - rhs.GetValue(i, j));
        }
    }
    return c;
}

Matrix Matrix::operator+(double rhs) const
{
    Matrix c = Matrix(row, column, false);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            c.SetValue(i, j, values.at(i).at(j) + rhs);
        }
    }
    return c;
}

Matrix Matrix::operator+(Matrix& rhs) const
{
    assert(row == rhs.row && column == rhs.column);
    Matrix c = Matrix(row, column, false);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            c.SetValue(i, j, values.at(i).at(j) + rhs.GetValue(i, j));
        }
    }
    return c;
}

Matrix& Matrix::operator-=(double& value)
{
    return *this = *this - value;
}

Matrix& Matrix::operator-=(Matrix& rhs)
{
    assert(row == rhs.row && column == rhs.column);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            values.at(i).at(j) = values.at(i).at(j) - rhs.GetValue(i, j);
        }
    }
    return *this;
}

Matrix Matrix::transpose()
{
    Matrix c = Matrix(column, row, false);
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            c.SetValue(j, i, values.at(i).at(j));
        }
    }
    return c;
}

std::vector<double> Matrix::Vectorize()
{
    std::vector<double> c;
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            c.emplace_back(values.at(i).at(j));
        }
    }
    return c;
}

void Matrix::SetValue(int row, int column, double value)
{
    values.at(row).at(column) = value;
}

double Matrix::GetValue(int row, int column)
{
    return values.at(row).at(column);
}

int Matrix::getrow()
{
    return row;
}

int Matrix::getcolumn()
{
    return column;
}

void Matrix::Print()
{
    for (int i = 0;i < row; i++)
    {
        for (int j = 0; j < column; j++)
        {
            std::cout << values.at(i).at(j) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}
