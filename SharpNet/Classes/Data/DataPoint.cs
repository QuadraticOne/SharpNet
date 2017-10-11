using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Data
{

    /// <summary>
    /// Stores supervised or unsupervised training examples for regression, classification, or
    /// unsupervised learning.
    /// </summary>
    public class DataPoint
    {

        public double[] input;
        /// <summary>
        /// The class to which the training point belongs.  A value of -1 denotes no class.
        /// </summary>
        public int category;
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
            category = -1;  // -1 denotes no class
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
            category = -1;  // -1 denotes no class
        }

        /// <summary>
        /// Assuming the data point has a category, sets the output to be a one-hot vector
        /// representing that category given the total number of categories.
        /// </summary>
        /// <param name="categories"></param>
        public void OneHot(int categories)
        {
            output = new double[categories];
            output[category] = 1.0;
        }

    }

}
