using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Maths
{

    /// <summary>
    /// Matrices, along with many common methods for manipulating them.
    /// </summary>
    public class Matrix
    {

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
                if ((row < 0) || (row >= Rows))
                    throw new IndexOutOfRangeException("Invalid row index (" + row + ").");
                if ((column < 0) || (column >= Columns))
                    throw new IndexOutOfRangeException("Invalid column index (" + column + ").");
                return matrix[row, column];
            }

            set
            {
                if ((row < 0) || (row >= Rows))
                    throw new IndexOutOfRangeException("Invalid row index (" + row + ").");
                if ((column < 0) || (column >= Columns))
                    throw new IndexOutOfRangeException("Invalid column index (" + column + ").");
                matrix[row, column] = value;
            }
        }

    }

}
