using SharpNet.Classes.Maths;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Maths
{

    /// <summary>
    /// Defines behaviour for activation functions, used to squash the outputs of a neuron to
    /// within a particular range.
    /// </summary>
    public abstract class ActivationFunction
    {

        private Matrix preActivation;

        /// <summary>
        /// Show the pre-activation vector to the activation function.  Any necessary work should
        /// be done here, such as finding the sum of the pre-activations, etc.  Calling the base
        /// method will save the pre-activation matrix to the variable `preActivation`.
        /// </summary>
        /// <param name="preActivation"></param>
        public virtual void Peek(Matrix preActivation)
        {
            this.preActivation = preActivation;
        }

        /// <summary>
        /// Calculate the value of the ith output given the ith element of the pre-activation
        /// vector.
        /// </summary>
        /// <param name="inputIndex"></param>
        /// <returns></returns>
        public abstract double Value(int inputIndex);

        /// <summary>
        /// Calculate the partial derivative of the given output index with respect to the given
        /// input index.  If outputs are independent of the pre-activation of other nodes, simply
        /// include `if (inputIndex != outputIndex) return 0`.
        /// </summary>
        /// <param name="inputIndex"></param>
        /// <param name="outputIndex"></param>
        /// <returns></returns>
        public abstract double Derivative(int inputIndex, int outputIndex);

        /// <summary>
        /// Calculate the output vector given a pre-activation vector.  This automatically calls
        /// Peek() on the pre-activation vector.
        /// </summary>
        /// <param name="preActivation"></param>
        /// <returns></returns>
        public Matrix Output(Matrix preActivation)
        {
            Peek(preActivation);
            Matrix m = new Matrix(preActivation.Rows, preActivation.Columns);

            for (int i = 0; i < m.Rows; i++) m[i, 0] = Value(i);

            return m;
        }

        /// <summary>
        /// Should return true iff the output of a node depends on the pre-activation of any node
        /// other than itself.  For example, for Softmax this is true, but for Relu this is false.
        /// </summary>
        /// <returns></returns>
        public abstract bool IsInterdependent();

        #region ACTIVATION_FUNCTIONS

        /// <summary>
        /// A continuous activation function which squashes inputs to the range (0, 1) and is
        /// approximately linear near the origin.
        /// </summary>
        public class Sigmoid : ActivationFunction
        {

            public override double Value(int inputIndex)
            {
                return 1 / (1 + Math.Exp(-preActivation[inputIndex, 0]));
            }

            public override double Derivative(int inputIndex, int outputIndex)
            {
                if (inputIndex != outputIndex) return 0;
                return Value(inputIndex) * (1 - Value(inputIndex));
            }

            public override bool IsInterdependent() => false;

        }

        /// <summary>
        /// Returns x if x is greater than 0, and 0 otherwise.
        /// </summary>
        public class Relu : ActivationFunction
        {

            public override double Value(int inputIndex)
            {
                if (preActivation[inputIndex, 0] > 0) return preActivation[inputIndex, 0];
                return 0;
            }

            public override double Derivative(int inputIndex, int outputIndex)
            {
                if (inputIndex != outputIndex) return 0;
                if (preActivation[inputIndex, 0] > 0) return 1;
                return 0;
            }

            public override bool IsInterdependent() => false;

        }

        /// <summary>
        /// Squashes each value to the range [0, 1] and ensures that the output vector sums to 1.
        /// </summary>
        public class Softmax : ActivationFunction
        {

            // The sum of the exponentiated values of the pre-activation vector
            private double sum = 0;

            public override void Peek(Matrix preActivation)
            {
                this.preActivation = preActivation;
                sum = 0;
                for (int i = 0; i < preActivation.Rows; i++) sum += Math.Exp(preActivation[i, 0]);
            }

            public override double Value(int inputIndex)
            {
                return Math.Exp(preActivation[inputIndex, 0]) / sum;
            }

            public override double Derivative(int inputIndex, int outputIndex)
            {
                if (inputIndex == outputIndex)
                {
                    return Value(inputIndex) * (1 - Value(inputIndex));  // &starthere - these always cancel out to zero, except rounding errors
                }
                else
                {
                    return -Value(inputIndex) * Value(outputIndex);
                }
            }

            public override bool IsInterdependent() => true;

        }

        #endregion  // ACTIVATION_FUNCTIONS

    }

}
