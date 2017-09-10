using SharpNet.Classes.Maths;
using SharpNet.Classes.Maths.Error;
using SharpNet.Classes.NeuralNetworkTrainer;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Architecture.NetworkLayer.Layers
{

    /// <summary>
    /// Defines a feedforward layer for a multilayer network.
    /// </summary>
    public abstract class FeedForwardLayer : Layer
    {

        public static int printed = 0;
        public static int printAt = 500;

        // TODO: override input setter

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
        /// Contains information for error backpropagation, allowing weight gradients to be
        /// calculated.
        /// </summary>
        public abstract class Gradient
        {

            protected FeedForwardLayer thisLayer = null;

            public Vector inputErrorDerivatives { get; private set; }
            protected Vector outputErrorDerivatives { get; private set; }
            /// <summary>
            /// Performs some setup which needs to occur for all gradient objects.  To be called at
            /// the end of a constructor.
            /// </summary>
            protected void Setup()
            {
                inputErrorDerivatives = new Vector(thisLayer.Inputs);
                outputErrorDerivatives = new Vector(thisLayer.Outputs);
            }

            /// <summary>
            /// Apply accumulated deltas to the target layer using the given learning rate, then
            /// clear any accumulated weight deltas.
            /// </summary>
            /// <param name="learningRate"></param>
            public virtual void ApplyDeltas(double learningRate) { }

            /// <summary>
            /// Apply accumulated deltas to the target layer using the given learning rate vector,
            /// then clear any accumulated weight deltas.
            /// </summary>
            /// <param name="learningRates"></param>
            public virtual void ApplyDeltas(Vector learningRates) { }

            /// <summary>
            /// Apply accumulated deltas to the target layer using the given learning rate matrix,
            /// then clear any accumulated weight deltas.
            /// </summary>
            /// <param name="learningRates"></param>
            public virtual void ApplyDeltas(Matrix learningRates) { }

            /// <summary>
            /// Update all error derivatives, and accumulate the appropriate weight deltas, given
            /// a reference to the gradient object of the next layer.
            /// </summary>
            /// <param name="nextLayerGradient"></param>
            public abstract void Backpropagate(Gradient nextLayerGradient,
                List<IRegulariser> regularisers);

            /// <summary>
            /// Update all error derivatives, and accumulate the appropriate weight deltas, given
            /// a reference to the loss function of the network and a target output, assuming that
            /// this is the output layer of the network.
            /// </summary>
            /// <param name="target"></param>
            /// <param name="lossFunction"></param>
            public abstract void Backpropagate(Matrix target, ILossFunction lossFunction,
                List<IRegulariser> regularisers);

        }

        /// <summary>
        /// Defines a fully connected layer, whose transfer function is defined as a matrix of
        /// weights which is multiplied by the input vector to produce a pre-activation.
        /// </summary>
        public class Dense : FeedForwardLayer
        {
            
            private ActivationFunction _activation;
            public ActivationFunction Activation
            {
                get { return _activation; }

                set
                {
                    outputIsAccurate = false;
                    _activation = value;
                }
            }

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
            /// Define a new fully connected feedforward layer.  The number of inputs does not
            /// include a bias term; this is added automatically.
            /// </summary>
            /// <param name="inputs"></param>
            /// <param name="outputs"></param>
            public Dense(int inputs, int outputs)
            {
                Inputs = inputs;
                Outputs = outputs;

                Input = new Matrix(Inputs, 1);  // Bias is added later
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
            /// Contains gradient information for a dense feedforward layer.
            /// </summary>
            public new class Gradient : FeedForwardLayer.Gradient
            {

                private Matrix weightDeltas;
                private Dense denseReference;  // Reference to the layer as a dense layer

                /// <summary>
                /// Set up the gradient object for a hidden layer.
                /// </summary>
                /// <param name="thisLayer"></param>
                /// <param name="nextLayerGradient"></param>
                public Gradient(Dense thisLayer)
                {
                    this.thisLayer = thisLayer;
                    denseReference = thisLayer;
                    weightDeltas = new Matrix(thisLayer.Weights.Rows, thisLayer.Weights.Columns);
                    Setup();
                }

                /// <summary>
                /// Apply accumulated deltas to the target layer using the given learning rate,
                /// then clear any accumulated weight deltas.
                /// </summary>
                /// <param name="learningRate"></param>
                public override void ApplyDeltas(double learningRate)
                {
                    denseReference.Weights = denseReference.Weights -
                        (learningRate * weightDeltas);

                    // &debug
                    if (printed == printAt) Console.WriteLine(weightDeltas.ToDetailedString());
                    printed++;

                    weightDeltas.Zero();
                    inputErrorDerivatives.Zero();
                    outputErrorDerivatives.Zero();
                }

                /// <summary>
                /// Apply accumulated deltas to the target layer using the given learning rate
                /// matrix, then clear any accumulated weight deltas.
                /// </summary>
                /// <param name="learningRates"></param>
                public override void ApplyDeltas(Matrix learningRates)
                {
                    Matrix piecewiseMultiply = learningRates.Copy();
                    for (int i = 0; i < piecewiseMultiply.Rows; i++)
                    {
                        for (int j = 0; j < piecewiseMultiply.Columns; j++)
                        {
                            piecewiseMultiply[i, j] = learningRates[i, j] * weightDeltas[i, j];
                        }
                    }
                    denseReference.Weights = denseReference.Weights - piecewiseMultiply;

                    weightDeltas.Zero();
                    inputErrorDerivatives.Zero();
                    outputErrorDerivatives.Zero();
                }

                /// <summary>
                /// Update all error derivatives, and accumulate the appropriate weight deltas,
                /// given a reference to the gradient object of the next layer.
                /// </summary>
                /// <param name="nextLayerGradient"></param>
                public override void Backpropagate(FeedForwardLayer.Gradient nextLayerGradient,
                    List<IRegulariser> regularisers)
                {
                    // Rate of change of output with respect to corresponding pre-activation
                    Vector preActivationOutputDerivatives = new Vector(thisLayer.Outputs);

                    for (int i = 0; i < thisLayer.Outputs; i++)
                    {
                        // Calculate output error derivatives; since this is a hidden layer, these
                        // are equal to the input error derivatives of the next layer
                        outputErrorDerivatives[i] = nextLayerGradient.inputErrorDerivatives[i];

                        // Calculate pre-activation derivatives
                        preActivationOutputDerivatives[i] = denseReference.Activation.Derivative(
                            denseReference.PreActivation[i, 0], i);
                    }

                    for (int i = 0; i < thisLayer.Inputs; i++)
                    {
                        // Calculate input error derivatives; the product of the derivative of the
                        // pre-activation wrt. the input and the derivative of the output wrt. the
                        // pre-activation, summed over all output neurons in this layer
                        double inputErrorDerivative = 0;
                        for (int j = 0; j < thisLayer.Outputs; j++)
                        {
                            inputErrorDerivative += denseReference.Weights[j, i] *
                                preActivationOutputDerivatives[j] * outputErrorDerivatives[j];
                        }
                        inputErrorDerivatives[i] = inputErrorDerivative;
                    }

                    // Cycle through each weight
                    for (int i = 0; i < denseReference.Weights.Rows; i++)  // i -> output neurons
                    {
                        for (int j = 0; j < denseReference.Weights.Columns; j++)
                            // j -> input neurons
                        {
                            double regulariserDelta = 0;
                            foreach (IRegulariser regulariser in regularisers)
                                regulariserDelta += regulariser.LossDerivative(
                                    denseReference.Weights[i, j]);

                            weightDeltas[i, j] = weightDeltas[i, j] +
                                outputErrorDerivatives[i] *
                                preActivationOutputDerivatives[i] *
                                thisLayer.Input[j, 0] + regulariserDelta;
                        }
                    }
                }

                /// <summary>
                /// Update all error derivatives, and accumulate the appropriate weight deltas,
                /// given a reference to the loss function of the network and a target output,
                /// assuming that this is the output layer of the network.
                /// </summary>
                /// <param name="target"></param>
                /// <param name="lossFunction"></param>
                public override void Backpropagate(Matrix target, ILossFunction lossFunction,
                    List<IRegulariser> regularisers)
                {
                    // Rate of change of output with respect to corresponding pre-activation
                    Vector preActivationOutputDerivatives = new Vector(thisLayer.Outputs);

                    for (int i = 0; i < thisLayer.Outputs; i++)
                    {
                        // Calculate output error derivatives; since this is an output layer, this
                        // is the error function derivative wrt. the ith output of this layer
                        outputErrorDerivatives[i] = lossFunction.ErrorDerivative(thisLayer.Output,
                            target, i);

                        // Calculate pre-activation derivatives
                        preActivationOutputDerivatives[i] = denseReference.Activation.Derivative(
                            denseReference.PreActivation[i, 0], i);
                    }

                    for (int i = 0; i < thisLayer.Inputs; i++)  // i -> input neuron
                    {
                        // Calculate input error derivatives; the product of the derivative of the
                        // pre-activation wrt. the input and the derivative of the output wrt. the
                        // pre-activation, summed over all output neurons in this layer
                        double inputErrorDerivative = 0;
                        for (int j = 0; j < thisLayer.Outputs; j++)  // j -> output neuron
                        {
                            inputErrorDerivative += denseReference.Weights[j, i] *
                                preActivationOutputDerivatives[j] * outputErrorDerivatives[j];
                        }
                        inputErrorDerivatives[i] = inputErrorDerivative;
                    }

                    // Cycle through each weight
                    for (int i = 0; i < denseReference.Weights.Rows; i++)  // i -> output neurons
                    {
                        for (int j = 0; j < denseReference.Weights.Columns; j++)
                        // j -> input neurons
                        {
                            double regulariserDelta = 0;
                            foreach (IRegulariser regulariser in regularisers)
                                regulariserDelta += regulariser.LossDerivative(
                                    denseReference.Weights[i, j]);

                            weightDeltas[i, j] = weightDeltas[i, j] +
                                outputErrorDerivatives[i] *
                                preActivationOutputDerivatives[i] *
                                thisLayer.Input[j, 0] + regulariserDelta;
                        }
                    }
                }

            }

        }

        /// <summary>
        /// Define a sparsely connected hidden layer.  Here, connections are stored as individual
        /// neurons rather than as a matrix of weights.
        /// </summary>
        public class Sparse : FeedForwardLayer
        {

            /// <summary>
            /// Define a new sparsely connected feedforward layer.  The number of inputs does not
            /// include a bias term; this is added automatically.
            /// </summary>
            /// <param name="inputs"></param>
            /// <param name="outputs"></param>
            public Sparse(int inputs, int outputs)
            {
                Inputs = inputs;
                Outputs = outputs;

                Input = new Matrix(Inputs, 1);  // Bias is added later
                Output = new Matrix(Outputs, 1);
            }

            /// <summary>
            /// Updates the output.
            /// </summary>
            protected override void UpdateOutput()
            {
                // Note when implementing: do not forget to set outputIsAccurate to true
                throw new NotImplementedException();
            }

            /// <summary>
            /// Contains gradient information for a sparse feedforward layer.
            /// </summary>
            public new class Gradient : FeedForwardLayer.Gradient
            {

                private Sparse sparseReference;  // Reference to the layer as a sparse layer

                /// <summary>
                /// Set up the gradient object for a hidden layer.
                /// </summary>
                /// <param name="thisLayer"></param>
                /// <param name="nextLayerGradient"></param>
                public Gradient(Sparse thisLayer)
                {
                    this.thisLayer = thisLayer;
                    sparseReference = thisLayer;
                    Setup();
                }

                /// <summary>
                /// Apply accumulated deltas to the target layer using the given learning rate,
                /// then clear any accumulated weight deltas.
                /// </summary>
                /// <param name="learningRate"></param>
                public override void ApplyDeltas(double learningRate)
                {
                    throw new NotImplementedException();
                }

                /// <summary>
                /// Update all error derivatives, and accumulate the appropriate weight deltas,
                /// given a reference to the gradient object of the next layer.
                /// </summary>
                /// <param name="nextLayerGradient"></param>
                public override void Backpropagate(FeedForwardLayer.Gradient nextLayerGradient,
                    List<IRegulariser> regularisers)
                {
                    throw new NotImplementedException();
                }

                /// <summary>
                /// Apply accumulated deltas to the target layer using the given learning rate
                /// vector, then clear any accumulated weight deltas.
                /// </summary>
                /// <param name="learningRates"></param>
                public override void ApplyDeltas(Vector learningRates)
                {
                    throw new NotImplementedException();
                }

                /// <summary>
                /// Update all error derivatives, and accumulate the appropriate weight deltas,
                /// given a reference to the loss function of the network and a target output,
                /// assuming that this is the output layer of the network.
                /// </summary>
                /// <param name="target"></param>
                /// <param name="lossFunction"></param>
                public override void Backpropagate(Matrix target, ILossFunction lossFunction,
                    List<IRegulariser> regularisers)
                {
                    throw new NotImplementedException();
                }

            }

        }

    }

}
