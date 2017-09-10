using SharpNet.Classes.Architecture.NetworkLayer.Layers;
using SharpNet.Classes.Maths;
using SharpNet.Classes.NeuralNetwork;
using SharpNet.Classes.NeuralNetwork.NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.NeuralNetworkTrainer
{

    /// <summary>
    /// Defines some commonly used initialisers.
    /// </summary>
    public static class Initialiser
    {

        /// <summary>
        /// Sets weights to be uniform, within a range.
        /// </summary>
        public class Uniform : IInitialiser
        {

            private double min, max;
            private bool zeroBias;

            /// <summary>
            /// Create a uniform initialiser, which samples weights from a uniform distribution
            /// ranging from min to max.
            /// </summary>
            /// <param name="min"></param>
            /// <param name="max"></param>
            public Uniform(double min, double max, bool zeroBias)
            {
                this.min = min;
                this.max = max;
                this.zeroBias = zeroBias;
            }

            /// <summary>
            /// Initialise the weights of a neural network.
            /// </summary>
            /// <param name="network"></param>
            public void Initialise(NeuralNetworkBase network)
            {
                if (network is FeedForwardNetwork)
                {
                    foreach (FeedForwardLayer layer in ((FeedForwardNetwork) network).Layers)
                    {
                        if (layer is FeedForwardLayer.Dense)
                        {
                            ((FeedForwardLayer.Dense)layer).Weights.RandomUniform(min, max);
                            if (zeroBias)
                            {
                                for (int i = 0; i < ((FeedForwardLayer.Dense)
                                    layer).Weights.Rows; i++)
                                {
                                    ((FeedForwardLayer.Dense)layer).Weights[i, 0] = 0;
                                }
                            }
                        }

                        if (layer is FeedForwardLayer.Sparse)
                        {
                            throw new NotImplementedException();
                        }
                    }
                }
            }

        }

        /// <summary>
        /// Allows the setting of custom weight matrices for easy testing.
        /// </summary>
        public class CustomTest : IInitialiser
        {

            private Matrix[] matrices;

            /// <summary>
            /// Create a new custom testing initialiser.
            /// </summary>
            /// <param name="matrices"></param>
            public CustomTest(Matrix[] matrices)
            {
                this.matrices = matrices;
            }

            /// <summary>
            /// Initialise the weights of a neural network.
            /// </summary>
            /// <param name="network"></param>
            public void Initialise(NeuralNetworkBase network)
            {
                FeedForwardNetwork feedForward = (FeedForwardNetwork)network;
                for (int i = 0; i < matrices.Length; i++)
                {
                    FeedForwardLayer.Dense layer = (FeedForwardLayer.Dense)feedForward.Layers[i];
                    layer.Weights = matrices[i];
                }
            }

        }

    }

}
