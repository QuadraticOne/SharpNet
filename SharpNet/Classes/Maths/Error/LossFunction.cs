using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Maths.Error
{

    /// <summary>
    /// Contains some commonly used loss functions.
    /// </summary>
    public class LossFunction
    {

        /// <summary>
        /// Loss is equal to half the magnitude of the difference between the output and target
        /// vectors.
        /// </summary>
        public class SquaredError : ILossFunction
        {

            /// <summary>
            /// Return the error between an output and a target.
            /// </summary>
            /// <param name="output"></param>
            /// <param name="target"></param>
            /// <returns></returns>
            public double Error(Matrix output, Matrix target)
            {
                Matrix difference = target - output;
                double error = 0;
                for (int i = 0; i < difference.Rows; i++)
                    error += difference[i, 0] * difference[i, 0];
                return 0.5 * error;
            }

            /// <summary>
            /// Return the derivative of the error with respect to the ith output.
            /// </summary>
            /// <param name="output"></param>
            /// <param name="target"></param>
            /// <param name="i"></param>
            /// <returns></returns>
            public double ErrorDerivative(Matrix output, Matrix target, int i)
            {
                return output[i, 0] - target[i, 0];
            }

        }

    }

}
