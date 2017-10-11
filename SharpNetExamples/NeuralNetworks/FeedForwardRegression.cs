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
    /// Example of a feedforward network being used for a simple regression problem.
    /// </summary>
    public class FeedForwardRegression
    {

        /// <summary>
        /// Create a feedforward neural network, train it, and run some test examples.
        /// </summary>
        public FeedForwardRegression()
        {

            // Define the function we are trying to model in the range [0, 1]

            double TestFunction(double x)
            {
                return (Math.Sqrt(x) + 0.3 * Math.Sin(6 * Math.Sqrt(x)));
            }

            // Create the data set

            // Frame the problem as a regression problem; one input (x) and one output (y)
            DataSet.Regression dataSet = new DataSet.Regression(1, 1);

            // Populate data set - data point takes a double array for inputs and outputs
            Random random = new Random(421);

            for (int i = 0; i < 10000; i++)
            {
                double nextDouble = random.NextDouble();  // In the range [0, 1]
                dataSet.AddDataPoint(new double[] { nextDouble },
                    new double[] { TestFunction(nextDouble) });
            }

            // Split the data points randomly into training, validation, and test sets in the ratio
            // 7:2:1
            dataSet.AssignDataPoints(0.7, 0.2, 0.1);

            // Create feedforward network

            // Create a network with one input and one output
            FeedForwardNetwork network = new FeedForwardNetwork(1, 1);

            // Add five hidden nodes, and connect them to the output layer using sigmoid activation
            network.AddHiddenLayer(32, new ActivationFunction.Sigmoid())
                .AddOutputLayer(new ActivationFunction.Sigmoid());

            // Set up the backpropagation trainer

            // Create the batch selector - here, select a random mini-batch from the training set
            // with 32 examples in it
            DataPoint[] select(DataSet data) => data.GetRandomTrainingSubset(32);

            BackpropagationTrainer trainer = new BackpropagationTrainer()
            {
                // Set the learning rate
                LearningRate = 0.1,

                // Initialise network weights using a uniform distribution
                initialiser = new Initialiser.Uniform(-0.01, 0.01, false),

                // Update weights after the whole batch rather than after each point
                stochastic = false,

                // Train on the whole training set each iteration
                batchSelector = new BackpropagationTrainer.BatchSelector(select),

                // Use squared error as the loss function
                lossFunction = new LossFunction.SquaredError(),

                // Log training data every 5000 epochs; access this with trainer.evaluations
                evaluationFrequency = 50
            };

            // Add a termination condition to the trainer; it will stop after 1200 epochs
            trainer.terminationConditions.Add(new TerminationCondition.EpochLimit(1200));

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

            // Print the test set loss
            Console.WriteLine("Mean test loss per point: " + trainer.TestLoss() + "\n");

            // Print 64 samples from the test set
            foreach (DataPoint dataPoint in dataSet.GetRandomTestSubset(64))
            {
                // Feed the network a test point and record the output
                double output = network.GetOutput(Matrix.ToColumnMatrix(dataPoint.input))[0, 0];
                Console.WriteLine("({0}) -> {1} : expected {2}",
                    dataPoint.input[0], output, dataPoint.output[0]);
            }
            Console.WriteLine();

            // If training takes a long time, this will notify you when it finishes
            Console.Beep(880, 2000);

        }

    }

}
