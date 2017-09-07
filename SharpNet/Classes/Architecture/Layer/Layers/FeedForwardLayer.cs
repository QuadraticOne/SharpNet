using SharpNet.Classes.Architecture.ActivationFunction;
using SharpNet.Classes.Maths;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Architecture.Layer.Layers
{

    /// <summary>
    /// Defines a fully connected feedforward of a multilayer network.
    /// </summary>
    public class FeedForwardLayer : Layer
    {

        private IActivationFunction _activation;
        public IActivationFunction Activation
        {
            get { return _activation; }

            set
            {
                outputIsAccurate = false;
                _activation = value;
            }
        }

        // TODO: override input setter

        private Matrix _preActivation;
        public Matrix PreActivation
        {
            get
            {
                if (!outputIsAccurate) UpdateOutput();
                return _preActivation;
            }

            protected set
            {
                outputIsAccurate = false;  // TODO: just set output to void
                _preActivation = value;
            }
        }

        // In the weight matrix, rows are outputs and columns are inputs
        private Matrix _weights;
        public Matrix Weights
        {
            get { return _weights; }

            set
            {
                _weights = value;
                outputIsAccurate = false;
            }
        }

        /// <summary>
        /// Define a new fully connected feedforward layer.  The number of inputs does not include
        /// a bias term; this is added automatically.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        public FeedForwardLayer(int inputs, int outputs)
        {
            Inputs = inputs;
            Outputs = outputs;

            Input = new Matrix(Inputs, 1);
            Output = new Matrix(Outputs, 1);

            Weights = new Matrix(Outputs, Inputs + 1);
        }

        /// <summary>
        /// Updates the output.
        /// </summary>
        protected override void UpdateOutput()
        {
            _preActivation = Weights * Input;
            _output = _preActivation.ApplyPiecewiseFunction(Activation.Value);
            outputIsAccurate = true;
        }

        /// <summary>
        /// Processes the given raw input into a form suitable for the layer to take directly.
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        protected override Matrix ProcessInput(Matrix matrix)
        {
            if ((matrix.Rows != Inputs) || (matrix.Columns != 1)) throw new ArgumentException(
                "The input matrix is not of the correct dimension.");
            Matrix newMatrix = new Matrix(Inputs + 1, 1);
            newMatrix[0, 0] = 1;
            for (int i = 0; i < matrix.Rows; i++) newMatrix[i + 1, 0] = matrix[i, 0];
            return newMatrix;
        }

        /// <summary>
        /// A fully connected feedforward layer which is almost identical to a regular one, but
        /// which allows the use of activation functions individual to each neuron, as opposed to a
        /// single activation function for the whole layer.
        /// </summary>
        public class IndividualActivations : FeedForwardLayer
        {

            // TODO: override update function

            private IActivationFunction[] activations;

            /// <summary>
            /// Instantiate a feedforward layer with individual activations.
            /// </summary>
            /// <param name="inputs"></param>
            /// <param name="outputs"></param>
            public IndividualActivations(int inputs, int outputs) : base(inputs, outputs)
            {
                Inputs = inputs;
                Outputs = outputs;

                activations = new IActivationFunction[Outputs];

                Input = new Matrix(Inputs, 1);
                Output = new Matrix(Outputs, 1);

                Weights = new Matrix(Outputs, Inputs + 1);
            }

        }

    }

}
