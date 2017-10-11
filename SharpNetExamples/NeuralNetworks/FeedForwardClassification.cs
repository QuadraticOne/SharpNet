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
    /// Example of a feedforward network being trained to perform a simple classification task.
    /// </summary>
    public class FeedForwardClassification
    {

        /// <summary>
        /// Create a feedforward neural network, train it, and run some test examples.
        /// </summary>
        public FeedForwardClassification()
        {

            // Create the data set

            // Create a data set for classification with two independent variables
            DataSet.Classification dataSet = new DataSet.Classification(2);

            // Populate data set - data point takes a double array for inputs and an integer class
            // as output
            Random random = new Random(386);

            for (int i = 0; i < 10000; i++)
            {
                double d1 = random.NextDouble();  // In the range [0, 1]
                double d2 = random.NextDouble();

                int category;
                if ((d1 < 0.5) && (d2 < 0.5)) category = 0;
                else if ((d1 < 0.5) && (d2 >= 0.5)) category = 1;
                else if (d2 < 0.5) category = 2;
                else category = 3;

                dataSet.AddDataPoint(new double[] { d1, d2 }, category);
            }

            dataSet.OneHotAll();  // Convert all data to one hot

            // Split the data points randomly into training, validation, and test sets in the ratio
            // 7:2:1
            dataSet.AssignDataPoints(0.7, 0.2, 0.1);
            dataSet.OneHotAll();

            // Create feedforward network

            // Create a network with two inputs and four outputs
            FeedForwardNetwork network = new FeedForwardNetwork(2, 4);

            // Add five hidden nodes with sigmoid activation, and connect them to the output layer
            // using softmax activation
            network.AddHiddenLayer(5, new ActivationFunction.Sigmoid())
                .AddOutputLayer(new ActivationFunction.Softmax());

            // Set up backpropagation trainer

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
                lossFunction = new LossFunction.NegativeLogProb(),

                // Log training data every 5000 epochs; access this with trainer.evaluations
                evaluationFrequency = 5
            };

            // Add a termination condition to the trainer; it will stop after 1200 epochs
            trainer.terminationConditions.Add(new TerminationCondition.EpochLimit(1));

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

        }

    }

}
