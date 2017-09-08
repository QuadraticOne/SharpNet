using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Maths
{

    /// <summary>
    /// Vectors, along with common methods for manipulating them.
    /// </summary>
    public class Vector
    {

        private static Random random = new Random();

        /// <summary>
        /// Elementwise addition of two vectors.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Vector operator+ (Vector a, Vector b)
        {
            if (a.Length != b.Length) throw new ArgumentException(
                "Cannot sum vectors of different lengths.");

            Vector newVector = new Vector(a.Length);
            for (int i = 0; i < newVector.Length; i++)
                newVector[i] = a[i] + b[i];
            return newVector;
        }

        /// <summary>
        /// Return the result of a vector multiplied by a scalar.
        /// </summary>
        /// <param name="vector"></param>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public static Vector operator* (Vector vector, double scalar)
        {
            Vector newVector = new Vector(vector.Length);
            for (int i = 0; i < newVector.Length; i++) newVector[i] = vector[i] * scalar;
            return newVector;
        }

        /// <summary>
        /// Return the result of a vector multiplied by a scalar.
        /// </summary>
        /// <param name="scalar"></param>
        /// <param name="vector"></param>
        /// <returns></returns>
        public static Vector operator* (double scalar, Vector vector)
        {
            return vector * scalar;
        }

        /// <summary>
        /// Elementwise subtraction of the right hand vector from the left hand vector.
        /// </summary>
        /// <param name="a"></param>
        /// <param name="b"></param>
        /// <returns></returns>
        public static Vector operator- (Vector a, Vector b)
        {
            if (a.Length != b.Length) throw new ArgumentException(
                "Cannot sum vectors of different lengths.");

            Vector newVector = new Vector(a.Length);
            for (int i = 0; i < newVector.Length; i++)
                newVector[i] = a[i] - b[i];
            return newVector;
        }

        public readonly int Length;
        private double[] vector;

        /// <summary>
        /// Create a new vector of specified length.
        /// </summary>
        /// <param name="length"></param>
        public Vector(int length)
        {
            Length = length;
            vector = new double[Length];
        }

        /// <summary>
        /// Create a new vector from a pre-existing array of doubles.
        /// </summary>
        /// <param name="vector"></param>
        public Vector(double[] vector)
        {
            Length = vector.Length;
            this.vector = vector;
        }

        /// <summary>
        /// Index the vectors.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public double this[int index]
        {
            get
            {
                ValidateIndexInput(index);
                return vector[index];
            }

            set
            {
                ValidateIndexInput(index);
                vector[index] = value;
            }
        }

        /// <summary>
        /// Return a copy of the vector.
        /// </summary>
        /// <returns></returns>
        public Vector Copy()
        {
            Vector newVector = new Vector(Length);
            for (int i = 0; i < Length; i++) newVector[i] = vector[i];
            return newVector;
        }

        /// <summary>
        /// Return the dot product of this vector with another.
        /// </summary>
        /// <param name="vector"></param>
        /// <returns></returns>
        public double Dot(Vector vector)
        {
            if (Length != vector.Length) throw new ArgumentException(
                "Cannot take the dot product of vectors of different lengths.");

            double dot = 0;
            for (int i = 0; i < vector.Length; i++) dot += this[i] * vector[i];
            return dot;
        }

        /// <summary>
        /// Set each element of the vector to 0.
        /// </summary>
        public void Zero()
        {
            for (int i = 0; i < Length; i++) vector[i] = 0;
        }

        /// <summary>
        /// Randomise the elements of the vector uniformly within a given range.
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        public void RandomUniform(double min, double max)
        {
            for (int i = 0; i < Length; i++)
            {
                this[i] = min + random.NextDouble() * (max - min);
            }
        }

        /// <summary>
        /// Randomise the elements of the vector according to a Gaussian distribution.
        /// </summary>
        /// <param name="mean"></param>
        /// <param name="standardDeviation"></param>
        public void RandomGaussian(double mean, double standardDeviation)
        {
            for (int i = 0; i < Length; i++)
            {
                // Sampling from a Gaussian requires uniform numbers in the range (0, 1], but
                // random.NextDouble() generates uniform numbers in [0, 1)
                double a = 1 - random.NextDouble();
                double b = 1 - random.NextDouble();

                double z = Math.Sqrt(-2.0 * Math.Log(a)) * Math.Cos(2.0 * Math.PI * b);
                this[i] = z * standardDeviation + mean;
            }
        }

        /// <summary>
        /// Return the vector in a string form.
        /// </summary>
        /// <returns></returns>
        new public string ToString()
        {
            string output = "[";

            for (int i = 0; i < Length; i++)
            {
                output += vector[i];
                if (i < Length - 1) output += ",";
            }

            output += "]";
            return output;
        }

        /// <summary>
        /// Return the vector in a detailed string form, suitable for printing to the console.
        /// </summary>
        /// <returns></returns>
        public string ToDetailedString()
        {
            string output = "[";

            for (int i = 0; i < Length; i++)
            {
                output += vector[i];
                if (i < Length - 1) output += ", ";
            }

            output += "]";
            return output;
        }

        /// <summary>
        /// Throw an error if the index is not within the correct range.
        /// </summary>
        /// <param name="index"></param>
        private void ValidateIndexInput(int index)
        {
            if ((index < 0) || (index >= Length)) throw new IndexOutOfRangeException(
                "Index is out of range.");
        }

    }

}
