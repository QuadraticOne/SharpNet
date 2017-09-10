﻿using System;
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
            DataSet.Regression ds = new DataSet.Regression(2, 1);
            ds.AddDataPoint(new double[] { 0, 0 }, new double[] { 0 });
            ds.AddDataPoint(new double[] { 0, 1 }, new double[] { 1 });
            ds.AddDataPoint(new double[] { 1, 0 }, new double[] { 1 });
            ds.AddDataPoint(new double[] { 1, 1 }, new double[] { 0 });
            //ds.AddDataPoint(new double[] { 0.2, 0.6 }, new double[] { 0.8 });
            ds.AssignDataPoints(1, 0, 0);

            Console.WriteLine("\nData set split in the ratio {0}, {1}, {2}.\n",
                ds.TrainingSet.Length, ds.ValidationSet.Length, ds.TestSet.Length);

            // Create neural net
            FeedForwardNetwork ffn = new FeedForwardNetwork(2, 1);
            ffn.AddHiddenLayer(5, new ActivationFunction.Sigmoid());
            ffn.AddOutputLayer(new ActivationFunction.Sigmoid());

            /*
            // Create weight matrices
            Matrix w0 = new Matrix(2, 3);
            w0[0, 0] = -0.3;
            w0[1, 0] = 0.1;
            w0[0, 1] = 0.4;
            w0[1, 1] = -0.2;
            w0[0, 2] = -0.6;
            w0[1, 2] = 0.3;

            Matrix w1 = new Matrix(1, 3);
            w1[0, 0] = -0.2;
            w1[0, 1] = 0.1;
            w1[0, 2] = 0.2;
            */

            // Create batch selector
            DataPoint[] select(DataSet data) => ds.TrainingSet;

            // Create backprop trainer
            BackpropagationTrainer t = new BackpropagationTrainer
            {
                stochastic = false,
                IndividualLearningRates = false,
                LearningRate = 0.15,
                initialiser = new Initialiser.Uniform(-0.2, 0.2, false),
                lossFunction = new LossFunction.SquaredError(),
                batchSelector = new BackpropagationTrainer.BatchSelector(select),
                evaluationFrequency = 10000
            };

            t.terminationConditions.Add(new TerminationCondition.EpochLimit(100000));
            t.Train(ffn, ds);

            // Troubleshoot trainer
            foreach (string s in t.Troubleshoot()) Console.WriteLine(s);

            Console.WriteLine();
            List<double[]> evals = t.evaluations;
            foreach (double[] arr in evals) Console.WriteLine("epoch={0}, training error={1}, " +
                "validation error={2}", arr[0], arr[1], arr[2]);
            Console.WriteLine();

            // Testing
            Random rand = new Random();
            int tests = 15;
            int correct = 0;
            for (int i = 0; i < tests; i++)
            {
                DataPoint dp = ds.GetRandomTrainingExample();
                double o = ffn.GetOutput(Matrix.ToColumnMatrix(dp.input))[0, 0];
                double oup = dp.output[0];
                Console.WriteLine(o + " " + oup);
                if (((o > 0.5) && (oup == 1)) || ((o < 0.5) && (oup == 0))) correct++;
            }

            Console.WriteLine("Tests correct: {0}/{1}", correct, tests);

            #endregion  // DEBUG_CODE

            Console.WriteLine("Debugging has finished.  Press ENTER to exit.");
            Console.Read();

        }

    }

}
