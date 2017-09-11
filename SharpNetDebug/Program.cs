using System;
using SharpNet.Classes.Maths;
using SharpNet.Classes.Architecture.NetworkLayer.Layers;
using SharpNet.Classes.NeuralNetwork.NeuralNetworks;
using SharpNet.Classes.Data;
using SharpNet.Classes.NeuralNetworkTrainer;
using SharpNet.Classes.Maths.Error;
using System.Collections.Generic;
using SharpNetExamples.NeuralNetworks;

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

            FeedForwardXor xor = new FeedForwardXor();

            #endregion  // DEBUG_CODE

            Console.WriteLine("Debugging has finished.  Press ENTER to exit.");
            Console.Read();

        }

    }

}
