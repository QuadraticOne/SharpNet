using SharpNet.Classes.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.NeuralNetworkTrainer
{

    /// <summary>
    /// Defines the behaviour of classes which regularise network weights during training.
    /// </summary>
    public interface IRegulariser
    {

        /// <summary>
        /// Calculate the loss of the network as calculated by this regulariser.
        /// </summary>
        /// <param name="network"></param>
        /// <returns></returns>
        double Loss(NeuralNetworkBase network);

        /// <summary>
        /// Calculate the change in the loss with respect to the change in a weight value.
        /// </summary>
        /// <param name="weight"></param>
        /// <returns></returns>
        double LossDerivative(double weight);

    }
}
