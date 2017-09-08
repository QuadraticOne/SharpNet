using System;
using SharpNet.Classes.Maths;
using SharpNet.Classes.Architecture.Layer.Layers;
using SharpNet.Classes.Architecture.ActivationFunction;
using SharpNet.Classes.NeuralNetwork.NeuralNetworks;

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
            FeedForwardNetwork ffn = new FeedForwardNetwork(3, 10);
            IActivationFunction sig = new ActivationFunction.Sigmoid();
            ffn.AddMultipleLayers(2, 8, sig).AddOutputLayer(sig);
            ffn.Input = new Matrix(3, 1);
            Console.WriteLine(ffn.Output.ToDetailedString());
            #endregion  // DEBUG_CODE

            Console.WriteLine("Debugging has finished.  Press ENTER to exit.");
            Console.Read();

        }

    }

}
