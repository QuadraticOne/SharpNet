using SharpNet.Classes.Maths;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Architecture.Layer.Layers
{

    /// <summary>
    /// Parent class which defines behaviour for neural network layers.  Specifically, a layer
    /// which inherits from this class takes a matrix of inputs with n rows and, by performing some
    /// process, produces an output matrix with m rows.
    /// </summary>
    public abstract class Layer
    {

        public int Inputs { get; protected set; }
        public int Outputs { get; protected set; }

        protected Matrix _input;
        public virtual Matrix Input {
            get
            {
                return _input;
            }

            set
            {
                _input = ProcessInput(value);
                outputIsAccurate = false;
            }
        }

        protected Matrix _output;
        public virtual Matrix Output
        {
            get
            {
                if (!outputIsAccurate)
                {
                    UpdateOutput();
                    outputIsAccurate = true;
                }
                return _output;
            }

            protected set
            {
                _output = value;
            }
        }

        protected bool outputIsAccurate;

        /// <summary>
        /// Give a new input to the layer, calculate the output, and return it.
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        public Matrix GetOutput(Matrix input)
        {
            Input = input;
            return this.Output;
        }

        /// <summary>
        /// Take an input matrix, check that it is a valid input matrix, perform any processing
        /// necessary to turn it into a valid matrix, and then return it.
        /// </summary>
        /// <param name="matrix"></param>
        protected abstract Matrix ProcessInput(Matrix matrix);

        /// <summary>
        /// Re-calculate the output vector and update the output field accordingly.
        /// </summary>
        protected abstract void UpdateOutput();

    }

}
