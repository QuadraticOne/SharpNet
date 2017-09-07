using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Architecture.ActivationFunction
{

    /// <summary>
    /// Defines behaviour for activation functions, used to squash the outputs of a neuron to
    /// within a particular range.
    /// </summary>
    public interface IActivationFunction
    {

        /// <summary>
        /// Return the value of the activation function at x.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        double Value(double x);

        /// <summary>
        /// Return the derivative of the activation function at x.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        double Derivative(double x);

    }
}
