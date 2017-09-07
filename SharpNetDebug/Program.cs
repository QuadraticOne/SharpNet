using System;
using SharpNet.Classes.Maths;
using SharpNet.Classes.Architecture.Layer.Layers;
using SharpNet.Classes.Architecture.ActivationFunction;

namespace SharpNetDebug
{

    /// <summary>
    /// Program for debugging SharpNet.  Does not contain any useful code.
    /// </summary>
    public class Program
    {

        /// <summary>
        /// Debugs SharpNet.  Code added here will be run when the project is run.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {

            Console.WriteLine("Debugging SharpNet.");

            #region DEBUG_CODE
            FeedForwardLayer ffl = new FeedForwardLayer(4, 4);

            ffl.Activation = new ActivationFunctions.Sigmoid();

            ffl.Weights.RandomUniform(-1, 1);
            Matrix m = new Matrix(4, 1);
            m.RandomUniform(-1, 1);
            ffl.Input = m;
            Console.WriteLine(ffl.Input.ToDetailedString());
            Console.WriteLine(ffl.Weights.ToDetailedString());

            Matrix pre = ffl.PreActivation;
            Console.WriteLine(pre.ToDetailedString());

            Matrix oup = ffl.Output;
            Console.WriteLine(oup.ToDetailedString());
            #endregion

            Console.WriteLine("Debugging has finished.  Press ENTER to exit.");
            Console.Read();

        }

    }

}
