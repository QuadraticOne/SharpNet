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

        /// <summary>
        /// Return the value of the activation function at x.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public abstract double Value(double x);

        /// <summary>
        /// Return the derivative of the activation function at x.
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public abstract double Derivative(double x);

        /// <summary>
        /// Show the activation function the pre-activation of the current row, allowing the
        /// activation function to calculate any necessary values from it.  This is useful for
        /// functions for which the output value or derivative may depend on the rest of the layer,
        /// such as the softmax function.
        /// </summary>
        /// <param name="m"></param>
        public virtual void SetPreActivation(Matrix m) { }  // Empty by default

        /// <summary>
        /// Return the activated value of the ith element in the pre-activation vector.  This is
        /// only necessary for functions which take the whole pre-activation vector as input.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="i"></param>
        /// <returns></returns>
        public virtual double Value(double x, int i)
        {
            return Value(x);
        }

        /// <summary>
        /// Return the derivative of the ith element in the pre-activation vector with respect to
        /// its output.  This is only necessary for functions which take the whole pre-activation
        /// vector as input.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="i"></param>
        /// <returns></returns>
        public virtual double Derivative(double x, int i)
        {
            return Derivative(x);
        }

        /// <summary>
        /// Maps inputs to the range (0, 1).  Approximately linear at the origin.
        /// </summary>
        public class Sigmoid : ActivationFunction
        {

            public override double Value(double x)
            {
                return 1 / (1 + Math.Exp(-x));
            }

            public override double Derivative(double x)
            {
                return Value(x) * (1 - Value(x));
            }

        }

        /// <summary>
        /// Returns x if x is greater than 0, and 0 otherwise.
        /// </summary>
        public class Relu : ActivationFunction
        {

            public override double Value(double x)
            {
                return Math.Max(0, x);
            }

            public override double Derivative(double x)
            {
                return (x < 0) ? 0 : 1;
            }

        }

    }

}
