using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.NeuralNetworkTrainer
{

    /// <summary>
    /// Defines the behaviour of classes which check to see whether a trainer should terminate its
    /// training.
    /// </summary>
    public interface ITerminationCondition
    {

        /// <summary>
        /// Return true if the trainer has finished.  Some termination conditions may require a
        /// specific kind of trainer, in which case a cast will be necessary.
        /// </summary>
        /// <param name="trainer"></param>
        /// <returns></returns>
        bool HasFinished(ITrainer trainer);

    }
}
