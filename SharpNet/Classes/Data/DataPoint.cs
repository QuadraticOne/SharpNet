using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Data
{

    /// <summary>
    /// Stores supervised or unsupervised training examples for regression, classification, or
    /// unsupervised learning.
    /// </summary>
    public struct DataPoint
    {

        public double[] input;
        public int? category;
        public double[] output;

        /// <summary>
        /// Constructor for regression training examples.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        public DataPoint(double[] input, double[] output)
        {
            this.input = input;
            this.output = output;
            category = null;
        }

        /// <summary>
        /// Constructor for classification training examples.
        /// </summary>
        /// <param name="input"></param>
        /// <param name="category"></param>
        public DataPoint(double[] input, int category)
        {
            this.input = input;
            output = null;
            this.category = category;
        }

        /// <summary>
        /// Constructor for unsupervised training examples.
        /// </summary>
        /// <param name="input"></param>
        public DataPoint(double[] input)
        {
            this.input = input;
            output = null;
            category = null;
        }

    }

}
