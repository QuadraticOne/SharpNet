using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Maths.Error
{

    /// <summary>
    /// Defines behaviour for loss functions.
    /// </summary>
    public interface ILossFunction
    {

        /// <summary>
        /// Calculate the error between the output vector of a matrix and the training target.
        /// </summary>
        /// <param name="output"></param>
        /// <param name="target"></param>
        /// <returns></returns>
        double Error(Matrix output, Matrix target);

        /// <summary>
        /// Calculate the change in error between the given network output and target with respect
        /// to the ith element.
        /// </summary>
        /// <param name="output"></param>
        /// <param name="target"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        double ErrorDerivative(Matrix output, Matrix target, int i);

    }

}
