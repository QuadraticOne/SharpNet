using SharpNet.Classes.Architecture.NetworkLayer.Layers;
using System;
using System.Collections.Generic;
using System.Text;
using SharpNet.Classes.Maths;

namespace SharpNet.Classes.NeuralNetwork.NeuralNetworks
{

    /// <summary>
    /// A feedforward network which deterministically and independently produces an output vector
    /// when given an input vector.
    /// </summary>
    public class FeedForwardNetwork : NeuralNetworkBase
    {

        private List<FeedForwardLayer> _layers = new List<FeedForwardLayer>();
        public List<FeedForwardLayer> Layers
        {
            get { return _layers; }
            private set { _layers = value; }
        }

        /// <summary>
        /// Set up a feedforward network with the number of inputs and outputs.  Then add layers to
        /// it using AddHiddenLayer() and AddOutputLayer().
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        public FeedForwardNetwork(int inputs, int outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }

        /// <summary>
        /// Add a hidden layer to the network, then returns the network.
        /// </summary>
        /// <param name="nodes"></param>
        /// <param name="activation"></param>
        /// <returns></returns>
        public FeedForwardNetwork AddHiddenLayer(int nodes, ActivationFunction activation)
        {
            _layers.Add(new FeedForwardLayer.Dense(
                (_layers.Count == 0 ? Inputs : _layers[_layers.Count - 1].Outputs), nodes)
            {
                Activation = activation
            });

            return this;
        }

        /// <summary>
        /// Adds an output layer to the network, then returns the network.
        /// </summary>
        /// <param name="activation"></param>
        /// <returns></returns>
        public FeedForwardNetwork AddOutputLayer(ActivationFunction activation)
        {
            _layers.Add(new FeedForwardLayer.Dense(_layers[_layers.Count - 1].Outputs,
                Outputs)
            {
                Activation = activation
            });

            return this;
        }

        /// <summary>
        /// Adds multiple hidden layers, each with the same number of nodes, to the network, then
        /// returns the network.
        /// </summary>
        /// <param name="layers"></param>
        /// <param name="nodesPerLayer"></param>
        /// <param name="activation"></param>
        /// <returns></returns>
        public FeedForwardNetwork AddMultipleLayers(int layers, int nodesPerLayer,
            ActivationFunction activation)
        {
            for (int i = 0; i < layers; i++)
            {
                AddHiddenLayer(nodesPerLayer, activation);
            }

            return this;
        }

        /// <summary>
        /// Update the output of the network so that it is consistent with the input.
        /// </summary>
        protected override void UpdateOutput()
        {
            _layers[0].Input = Input;
            for (int i = 1; i < _layers.Count; i++)
            {
                _layers[i].Input = _layers[i - 1].Output;
            }
            Output = _layers[_layers.Count - 1].Output;
        }

    }

}
