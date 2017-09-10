﻿using SharpNet.Classes.Architecture.NetworkLayer.Layers;
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

            /// <summary>
            /// Create a uniform initialiser, which samples weights from a uniform distribution
            /// ranging from min to max.
            /// </summary>
            /// <param name="min"></param>
            /// <param name="max"></param>
            public Uniform(double min, double max)
            {
                this.min = min;
                this.max = max;
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
                        }

                        if (layer is FeedForwardLayer.Sparse)
                        {
                            throw new NotImplementedException();
                        }
                    }
                }
            }

        }

    }

}
