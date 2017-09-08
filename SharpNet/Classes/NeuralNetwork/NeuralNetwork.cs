using SharpNet.Classes.Maths;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.NeuralNetwork
{

    /// <summary>
    /// Base class from which all neural networks inherit.  The primary function of a neural
    /// network is to take an input vector, and convert it to an output vector by some transform
    /// function.
    /// </summary>
    public abstract class NeuralNetwork
    {

        public int Inputs { get; protected set; }
        public int Outputs { get; protected set; }

        protected Matrix _input;
        public Matrix Input
        {
            get { return _input; }

            set
            {
                if ((value.Rows != Inputs) || (value.Columns != 1)) throw new ArgumentException(
                    "The structure of this input is not compatible with the network.");

                _input = value;
                UpdateOutput();  // Update output immediately, for backpropagation
            }
        }

        protected Matrix _output;
        public Matrix Output
        {
            get
            {
                if (_output == null) UpdateOutput();
                return _output;
            }

            protected set { _output = value; }
        }

        /// <summary>
        /// Get the output corresponding to a given input.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix GetOutput(Matrix input)
        {
            Input = input;
            return Output;
        }

        /// <summary>
        /// Update the output of the network so that it is valid, given the input.
        /// </summary>
        protected abstract void UpdateOutput();

    }

}
