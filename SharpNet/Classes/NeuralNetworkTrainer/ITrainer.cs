using SharpNet.Classes.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.NeuralNetworkTrainer
{

    /// <summary>
    /// Defines behaviour for a class which trains neural networks.
    /// </summary>
    public interface ITrainer
    {

        /// <summary>
        /// Return the current epoch count of the trainer.
        /// </summary>
        /// <returns></returns>
        int GetEpoch();

        /// <summary>
        /// Calculate and return the network loss on a specific example.
        /// </summary>
        /// <param name="example"></param>
        /// <returns></returns>
        double Loss(DataPoint example);

        /// <summary>
        /// Evaluate the mean training loss.
        /// </summary>
        /// <returns></returns>
        double TrainingLoss();

        /// <summary>
        /// Evaluate the mean validation loss.
        /// </summary>
        /// <returns></returns>
        double ValidationLoss();

    }
}
