using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.NeuralNetworkTrainer
{

    /// <summary>
    /// Defines some commonly used training termination conditions.
    /// </summary>
    public class TerminationCondition
    {

        /// <summary>
        /// Stops training after a certain number of epochs have occurred.
        /// </summary>
        public class EpochLimit : ITerminationCondition
        {

            private int limit;

            /// <summary>
            /// Create a new epoch limit termination condition.
            /// </summary>
            /// <param name="limit"></param>
            public EpochLimit(int limit)
            {
                this.limit = limit;
            }

            public bool HasFinished(ITrainer trainer)
            {
                return trainer.GetEpoch() > limit;
            }

        }

    }

}
