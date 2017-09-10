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

        public static int printed = 0;

        /// <summary>
        /// Debugs SharpNet.  Code added here will be run when the project is run.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {

            Console.WriteLine("Debugging SharpNet.");

            #region DEBUG_CODE

            // Create data set
            DataSet.Regression ds = new DataSet.Regression(2, 1);  // XOR
            ds.AddDataPoint(new double[] { 0, 0 }, new double[] { 0 });
            ds.AddDataPoint(new double[] { 1, 0 }, new double[] { 1 });
            ds.AddDataPoint(new double[] { 0, 1 }, new double[] { 1 });
            ds.AddDataPoint(new double[] { 1, 1 }, new double[] { 0 });
            Random rand = new Random();
            /*for (int i = 0; i < 10000; i++)
            {
                double d = rand.NextDouble();
                ds.AddDataPoint(new double[] { d }, new double[] { d });
            }*/
            ds.AssignDataPoints(1, 0, 0);
            Console.WriteLine("\nData set split in the ratio {0}, {1}, {2}.\n",
                ds.TrainingSet.Length, ds.ValidationSet.Length, ds.TestSet.Length);

            // Create neural net
            FeedForwardNetwork ffn = new FeedForwardNetwork(2, 1);
            ffn.AddHiddenLayer(4, new ActivationFunction.Sigmoid());
            ffn.AddOutputLayer(new ActivationFunction.Sigmoid());

            // Create batch selector
            DataPoint[] select(DataSet data) => ds.GetRandomTrainingSubset(4);

            // Create backprop trainer
            BackpropagationTrainer t = new BackpropagationTrainer
            {
                stochastic = false,
                IndividualLearningRates = false,
                LearningRate = 0.005,
                initialiser = new Initialiser.Uniform(-0.2, 0.2, false),
                lossFunction = new LossFunction.SquaredError(),
                batchSelector = new BackpropagationTrainer.BatchSelector(select),
                evaluationFrequency = 2000
            };

            t.terminationConditions.Add(new TerminationCondition.EpochLimit(50000));
            t.Train(ffn, ds);

            // Troubleshoot trainer
            foreach (string s in t.Troubleshoot()) Console.WriteLine(s);

            Console.WriteLine();
            List<double[]> evals = t.evaluations;
            foreach (double[] arr in evals) Console.WriteLine("epoch={0}, training error={1}, " +
                "validation error={2}", arr[0], arr[1], arr[2]);
            Console.WriteLine();

            // Manual tests
            for (int i = 0; i < 1; i++)
            {
                Matrix m = new Matrix(2, 1);
                m[0, 0] = (rand.NextDouble() > 0.5) ? 1 : 0;
                m[1, 0] = (rand.NextDouble() > 0.5) ? 1 : 0;
                ffn.Input = m;
                Console.WriteLine(ffn.Layers[0].Input.ToDetailedString());
                Console.WriteLine(((FeedForwardLayer.Dense)ffn.Layers[0]).Weights.ToDetailedString());
                Console.WriteLine(ffn.Layers[0].Output.ToDetailedString());
                Console.WriteLine(ffn.Layers[1].Input.ToDetailedString());
                Console.WriteLine(((FeedForwardLayer.Dense)ffn.Layers[1]).Weights.ToDetailedString());
                Console.WriteLine(ffn.Layers[1].Output.ToDetailedString());
            }

            Console.WriteLine();

            //Console.WriteLine("Backprop trainer ready?  {0}", t.IsReady());

            #endregion  // DEBUG_CODE

            Console.WriteLine("Debugging has finished.  Press ENTER to exit.");
            Console.Read();

        }

    }

}
