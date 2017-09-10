using System;
using SharpNet.Classes.Maths;
using SharpNet.Classes.Architecture.NetworkLayer.Layers;
using SharpNet.Classes.NeuralNetwork.NeuralNetworks;
using SharpNet.Classes.Data;
using SharpNet.Classes.NeuralNetworkTrainer;
using SharpNet.Classes.Maths.Error;
using System.Collections.Generic;

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

            // Create data set
            DataSet.Regression ds = new DataSet.Regression(1, 1);  // Models the function f(x) = x, x ~ [0, 1]
            Random rand = new Random();
            for (int i = 0; i < 10000; i++)
            {
                double d = rand.NextDouble();
                ds.AddDataPoint(new double[] { d }, new double[] { d });
            }
            ds.AssignDataPoints(0.7, 0.2, 0.1);
            Console.WriteLine("\nData set split in the ratio {0}, {1}, {2}.\n",
                ds.TrainingSet.Length, ds.ValidationSet.Length, ds.TestSet.Length);

            // Create neural net
            FeedForwardNetwork ffn = new FeedForwardNetwork(1, 1);
            ffn.AddHiddenLayer(8, new ActivationFunction.Sigmoid());
            ffn.AddHiddenLayer(8, new ActivationFunction.Sigmoid());
            ffn.AddOutputLayer(new ActivationFunction.Sigmoid());

            // Create batch selector
            DataPoint[] select(DataSet data) => ds.GetRandomTrainingSubset(32);

            // Create backprop trainer
            BackpropagationTrainer t = new BackpropagationTrainer
            {
                IndividualLearningRates = false,
                LearningRate = 0.01,
                initialiser = new Initialiser.Uniform(-0.5, 0.5),
                lossFunction = new LossFunction.SquaredError(),
                batchSelector = new BackpropagationTrainer.BatchSelector(select),
                evaluationFrequency = 1
            };

            t.terminationConditions.Add(new TerminationCondition.EpochLimit(100));
            t.Train(ffn, ds);

            // Troubleshoot trainer
            foreach (string s in t.Troubleshoot()) Console.WriteLine(s);

            Console.WriteLine();
            List<double[]> evals = t.evaluations;
            foreach (double[] arr in evals) Console.WriteLine("epoch={0}, training error={1}, " +
                "validation error={2}", arr[0], arr[1], arr[2]);
            Console.WriteLine();

            // Manual tests
            for (int i = 0; i < 10; i++)
            {
                Matrix m = new Matrix(1, 1);
                m[0, 0] = 0.1 * i;
                ffn.Input = m;
                Console.WriteLine(ffn.Output.ToDetailedString());
            }

            Console.WriteLine();

            //Console.WriteLine("Backprop trainer ready?  {0}", t.IsReady());

            #endregion  // DEBUG_CODE

            Console.WriteLine("Debugging has finished.  Press ENTER to exit.");
            Console.Read();

        }

    }

}
