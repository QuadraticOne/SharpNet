using System;
using System.Collections.Generic;
using System.Text;
using SharpNet.Classes.NeuralNetwork;

namespace SharpNet.Classes.NeuralNetworkTrainer
{

    /// <summary>
    /// Defines behaviour for an initialiser, which sets the initial weights of a neural network.
    /// </summary>
    public interface IInitialiser
    {

        /// <summary>
        /// Initialise the weights of the neural network.
        /// </summary>
        /// <param name="network"></param>
        void Initialise(NeuralNetworkBase network);

    }

}
