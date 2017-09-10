using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Maths
{

    /// <summary>
    /// Matrices, along with many common methods for manipulating them.
    /// </summary>
    public class Matrix
    {

        /// <summary>
        /// Return a column matrix whose elements are equal to those of an array.
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public static Matrix ToColumnMatrix(double[] array)
        {
            Matrix matrix = new Matrix(array.Length, 1);
            for (int i = 0; i < array.Length; i++) matrix[i, 0] = array[i];
            return matrix;
        }

        /// <summary>
        /// Return a row matrix whose elements are equal to those of an array.
        /// </summary>
        /// <param name="array"></param>
        /// <returns></returns>
        public static Matrix ToRowMatrix(double[] array)
        {
            Matrix matrix = new Matrix(1, array.Length);
            for (int i = 0; i < array.Length; i++) matrix[0, 1] = array[i];
            return matrix;
        }

        private static Random random = new Random();

        /// <summary>
        /// Elementwise addition of two matrices.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Matrix operator+ (Matrix a, Matrix b)
        {
            if (a.Rows != b.Rows) throw new ArgumentException(
                "Summed matrices must have the same number of rows.");
            if (a.Columns != b.Columns) throw new ArgumentException(
                "Summed matrices must have the same number of columns.");

            Matrix matrix = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    matrix[i, j] = a[i, j] + b[i, j];
                }
            }
            return matrix;
        }

        /// <summary>
        /// Subtract the matrix on the right from the matrix on the left.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Matrix operator- (Matrix a, Matrix b)
        {
            if (a.Rows != b.Rows) throw new ArgumentException(
                "Subtracted matrices must have the same number of rows.");
            if (a.Columns != b.Columns) throw new ArgumentException(
                "Subtracted matrices must have the same number of columns.");

            Matrix matrix = new Matrix(a.Rows, a.Columns);
            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    matrix[i, j] = a[i, j] - b[i, j];
                }
            }
            return matrix;
        }

        /// <summary>
        /// Multiply each element of a matrix by a scalar.
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public static Matrix operator* (Matrix matrix, double scalar)
        {
            Matrix newMatrix = new Matrix(matrix.Rows, matrix.Columns);
            for (int i = 0; i < newMatrix.Rows; i++)
            {
                for (int j = 0; j < newMatrix.Columns; j++)
                {
                    newMatrix[i, j] = matrix[i, j] * scalar;
                }
            }
            return newMatrix;
        }

        /// <summary>
        /// Multiply each element of a matrix by a scalar.
        /// </summary>
        /// <param name="scalar"></param>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public static Matrix operator* (double scalar, Matrix matrix)
        {
            return matrix * scalar;
        }

        /// <summary>
        /// Multiply the matrix on the left by the matrix on the right.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Matrix operator* (Matrix a, Matrix b)
        {
            if (a.Columns != b.Rows) throw new ArgumentException(
                "Matrices of these sizes cannot be multiplied.");

            Matrix matrix = new Matrix(a.Rows, b.Columns);

            for (int i = 0; i < matrix.Rows; i++)
            {
                for (int j = 0; j < matrix.Columns; j++)
                {
                    double dot = 0;
                    for (int k = 0; k < a.Columns; k++)
                    {
                        dot += a[i, j] * b[k, j];
                    }
                    matrix[i, j] = dot;
                }
            }

            return matrix;
        }

        public readonly int Rows, Columns;
        private double[,] matrix;

        /// <summary>
        /// Create a matrix of the given size with all elements initialised to zero.
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="columns"></param>
        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;
            matrix = new double[Rows, Columns];
        }

        /// <summary>
        /// Create a matrix around a 2D array of doubles.
        /// </summary>
        /// <param name="matrix"></param>
        public Matrix(double[,] matrix)
        {
            Rows = matrix.GetLength(0);
            Columns = matrix.GetLength(1);
            this.matrix = matrix;
        }

        /// <summary>
        /// Index the matrix.
        /// </summary>
        /// <param name="column"></param>
        /// <param name="row"></param>
        /// <returns></returns>
        public double this[int row, int column]
        {
            get
            {
                ValidateRowInput(row);
                ValidateColumnInput(column);
                return matrix[row, column];
            }

            set
            {
                ValidateRowInput(row);
                ValidateColumnInput(column);
                matrix[row, column] = value;
            }
        }

        /// <summary>
        /// Returns true if the matrix is a square matrix.
        /// </summary>
        /// <returns></returns>
        public bool IsSquare()
        {
            return (Rows == Columns);
        }

        /// <summary>
        /// Copy this matrix.
        /// </summary>
        /// <returns></returns>
        public Matrix Copy()
        {
            Matrix newMatrix = new Matrix(Rows, Columns);

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++) newMatrix[i, j] = matrix[i, j];
            }

            return newMatrix;
        }

        /// <summary>
        /// Returns the selected row from the matrix.
        /// </summary>
        /// <param name="row"></param>
        /// <returns></returns>
        public Matrix GetRow(int row)
        {
            ValidateRowInput(row);

            Matrix matrixRow = new Matrix(1, Columns);
            for (int j = 0; j < Columns; j++) matrixRow[0, j] = matrix[row, j];
            return matrixRow;
        }

        /// <summary>
        /// Returns the selected column from the matrix.
        /// </summary>
        /// <param name="column"></param>
        /// <returns></returns>
        public Matrix GetColumn(int column)
        {
            ValidateColumnInput(column);

            Matrix matrixColumn = new Matrix(Rows, 1);
            for (int i = 0; i < Rows; i++) matrixColumn[i, 0] = matrix[i, column];
            return matrixColumn;
        }

        /// <summary>
        /// Returns a sub-matrix from this matrix, starting at the given coordinates and working
        /// down and to the right.
        /// </summary>
        /// <param name="rowStart"></param>
        /// <param name="columnStart"></param>
        /// <param name="width"></param>
        /// <param name="height"></param>
        /// <returns></returns>
        public Matrix GetSubMatrix(int rowStart, int columnStart, int width, int height)
        {
            ValidateRowInput(rowStart);
            ValidateColumnInput(columnStart);

            // TODO: add better argument validation

            Matrix subMatrix = new Matrix(width, height);

            for (int i = rowStart; i < rowStart + width; i++)
            {
                for (int j = columnStart; j < columnStart + height; j++)
                {
                    subMatrix[i - rowStart, j - columnStart] = matrix[i, j];
                }
            }

            return subMatrix;
        }

        /// <summary>
        /// Return a copy of the matrix with a row removed.
        /// </summary>
        /// <param name="row"></param>
        /// <returns></returns>
        public Matrix RemoveRow(int row)
        {
            Matrix newMatrix = new Matrix(Rows - 1, Columns);

            int copyRowDelta = 0;
            for (int i = 0; i < Rows - 1; i++)
            {
                if (i == row) copyRowDelta = 1;
                for (int j = 0; j < Columns; j++)
                {
                    newMatrix[i, j] = matrix[i + copyRowDelta, j];
                }
            }

            return newMatrix;
        }

        /// <summary>
        /// Return a copy of the matrix with a row removed.
        /// </summary>
        /// <param name="column"></param>
        /// <returns></returns>
        public Matrix RemoveColumn(int column)
        {
            Matrix newMatrix = new Matrix(Rows, Columns - 1);

            int copyColumnDelta = 0;
            for (int j = 0; j < Columns - 1; j++)
            {
                if (j == column) copyColumnDelta = 1;
                for (int i = 0; i < Rows; i++)
                {
                    newMatrix[i, j] = matrix[i, j + copyColumnDelta];
                }
            }

            return newMatrix;
        }

        /// <summary>
        /// Calculate and return the determinant of the matrix.  Note that this is exceedingly
        /// computationally expensive for large matrices.
        /// </summary>
        /// <returns></returns>
        public double Determinant()
        {
            // TODO: implement an algorithm better than O(n!)

            if (!IsSquare()) throw new ArgumentException(
                "Cannot calculate the determinant of a non-square matrix.");

            if (Rows == 2)
            {
                return matrix[0, 0] * matrix[1, 1] - matrix[1, 0] * matrix[0, 1];
            }
            else
            {
                double determinant = 0;

                for (int j = 0; j < Columns; j++)
                {
                    determinant += matrix[0, j] *
                        ((j % 2 == 0) ? 1 : 0) *
                        this.RemoveRow(0).RemoveColumn(j).Determinant();
                }

                return determinant;
            }
        }

        /// <summary>
        /// Update the value of each element of the matrix according to a function, which takes the
        /// value of that element as input.
        /// </summary>
        /// <param name="function"></param>
        /// <returns></returns>
        public Matrix ApplyPiecewiseFunction(Func<double, double> function)
        {
            Matrix newMatrix = new Matrix(Rows, Columns);
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    newMatrix[i, j] = function(matrix[i, j]);
                }
            }
            return newMatrix;
        }

        /// <summary>
        /// Zero the elements of the matrix.
        /// </summary>
        public void Zero()
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++) matrix[i, j] = 0;
            }
        }

        /// <summary>
        /// Randomise the elements of the matrix uniformly within a given range.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        public void RandomUniform(double min, double max)
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    this[i, j] = min + random.NextDouble() * (max - min);
                }
            }
        }

        /// <summary>
        /// Randomise the elements of the matrix according to a Gaussian distribution.
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="standardDeviation"></param>
        public void RandomGaussian(double mean, double standardDeviation)
        {
            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    // Sampling from a Gaussian requires uniform numbers in the range (0, 1], but
                    // random.NextDouble() generates uniform numbers in [0, 1)
                    double a = 1 - random.NextDouble();
                    double b = 1 - random.NextDouble();

                    double z = Math.Sqrt(-2.0 * Math.Log(a)) * Math.Cos(2.0 * Math.PI * b);
                    this[i, j] = z * standardDeviation + mean;
                }
            }
        }

        /// <summary>
        /// Returns the string form of the matrix.
        /// </summary>
        /// <returns></returns>
        new public string ToString()
        {
            string output = "[";

            for (int i = 0; i < Rows; i++)
            {
                output += "[";
                for (int j = 0; j < Columns; j++)
                {
                    output += matrix[i, j];
                    if (j < Columns - 1) output += ",";
                }
                output += "]";
                if (i < Rows - 1) output += ",";
            }
            output += "]";

            return output;
        }

        /// <summary>
        /// Returns the matrix in a more detailed string form, suitable to log to the console.
        /// </summary>
        /// <returns></returns>
        public string ToDetailedString()
        {
            string output = "[";

            for (int i = 0; i < Rows; i++)
            {
                output += "\n  [";
                for (int j = 0; j < Columns; j++)
                {
                    output += matrix[i, j];
                    if (j < Columns - 1) output += ", ";
                }
                output += "]";
            }
            output += "\n]";

            return output;
        }

        /// <summary>
        /// Throws an error if the row input does not exist.
        /// </summary>
        /// <param name="row"></param>
        private void ValidateRowInput(int row)
        {
            if ((row < 0) || (row >= Rows))
                throw new IndexOutOfRangeException("Invalid row index (" + row + ").");
        }

        /// <summary>
        /// Throws an error if the column input does not exist.
        /// </summary>
        /// <param name="column"></param>
        private void ValidateColumnInput(int column)
        {
            if ((column < 0) || (column >= Columns))
                throw new IndexOutOfRangeException("Invalid column index (" + column + ").");
        }

    }

}
