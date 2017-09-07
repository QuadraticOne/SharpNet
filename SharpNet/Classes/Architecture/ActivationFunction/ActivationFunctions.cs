using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Architecture.ActivationFunction
{

    /// <summary>
    /// Contains some commonly used activation functions.
    /// </summary>
    public static class ActivationFunctions
    {

        /// <summary>
        /// Maps inputs to the range (0, 1).  Approximately linear at the origin.
        /// </summary>
        public class Sigmoid : IActivationFunction
        {

            public double Value(double x)
            {
                return 1 / (1 + Math.Exp(-x));
            }

            public double Derivative(double x)
            {
                return Value(x) * (1 - Value(x));
            }

        }

    }

}
