using SharpNet.Classes.Data;
using SharpNet.Classes.Maths;
using SharpNet.Classes.Maths.Error;
using SharpNet.Classes.NeuralNetwork.NeuralNetworks;
using SharpNet.Classes.NeuralNetworkTrainer;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SharpNetExamples.NeuralNetworks
{

    /// <summary>
    /// Example of creating a feedforward neural network and training it to classify an XOR gate.
    /// </summary>
    public class FeedForwardXor
    {

        /// <summary>
        /// Create a feedforward neural network, train it, and run some test examples.
        /// </summary>
        public FeedForwardXor()
        {

            // Create the data set

            // Frame the problem as a regression problem; two inputs (x, y) and one output (z)
            DataSet.Regression dataSet = new DataSet.Regression(2, 1);

            // Populate data set - data point takes a double array for inputs and outputs
            dataSet.AddDataPoint(new double[] { 0, 0 }, new double[] { 0 });
            dataSet.AddDataPoint(new double[] { 1, 0 }, new double[] { 1 });
            dataSet.AddDataPoint(new double[] { 0, 1 }, new double[] { 1 });
            dataSet.AddDataPoint(new double[] { 1, 1 }, new double[] { 0 });

            // Split the data points randomly into training, validation, and test sets
            // Here, put all data points into the training set
            dataSet.AssignDataPoints(1, 0, 0);

            // Create feedforward network

            // Create a network with two inputs and one output
            FeedForwardNetwork network = new FeedForwardNetwork(2, 1);

            // Add five hidden nodes, and connect them to the output layer using sigmoid activation
            network.AddHiddenLayer(5, new ActivationFunction.Sigmoid())
                .AddOutputLayer(new ActivationFunction.Sigmoid());

            // Set up the backpropagation trainer

            // Create the batch selector - here, select the whole training set, but this can be
            // used to select mini-batches, single points, etc.
            DataPoint[] select(DataSet data) => data.TrainingSet;

            BackpropagationTrainer trainer = new BackpropagationTrainer()
            {
                // Set the learning rate
                LearningRate = 0.07,

                // Initialise network weights using a uniform distribution
                initialiser = new Initialiser.Uniform(-0.2, 0.2, false),

                // Update weights after the whole batch rather than after each point
                stochastic = false,

                // Train on the whole training set each iteration
                batchSelector = new BackpropagationTrainer.BatchSelector(select),

                // Use squared error as the loss function
                lossFunction = new LossFunction.SquaredError(),

                // Log training data every 5000 epochs; access this with trainer.evaluations
                evaluationFrequency = 5000
            };

            // Add a termination condition to the trainer; it will stop after 50,000 epochs
            trainer.terminationConditions.Add(new TerminationCondition.EpochLimit(50000));

            // Troubleshoot trainer (this will notify you of any missing required settings)
            foreach (string s in trainer.Troubleshoot()) Console.WriteLine();

            // Start training

            // Train the network on the data set
            trainer.Train(network, dataSet);

            // Print training data to the console
            List<double[]> evals = trainer.evaluations;
            foreach (double[] arr in evals) Console.WriteLine("epoch={0}, training error={1}, " +
                "validation error={2}", arr[0], arr[1], arr[2]);
            Console.WriteLine();

            // Test

            foreach (DataPoint dataPoint in dataSet.TrainingSet)
            {
                // Feed the network a test point and record the output
                double output = network.GetOutput(Matrix.ToColumnMatrix(dataPoint.input))[0, 0];
                Console.WriteLine("({0}, {1}) -> {2} : expected {3}",
                    dataPoint.input[0], dataPoint.input[1], output, dataPoint.output[0]);
            }
            Console.WriteLine();

        }

    }
}
