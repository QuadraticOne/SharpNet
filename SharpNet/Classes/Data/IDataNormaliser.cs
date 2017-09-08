using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Data
{

    /// <summary>
    /// Defines the behaviour of data normalising classes.
    /// </summary>
    public interface IDataNormaliser
    {

        /// <summary>
        /// Perform any fitting necessary (for example, calculating mean and standard deviation) to
        /// a data set.
        /// </summary>
        /// <param name="dataSet"></param>
        void Fit(DataPoint[] dataSet);

        /// <summary>
        /// Return true if fitting has occurred; false otherwise.
        /// </summary>
        /// <returns></returns>
        bool HasBeenFit();

        /// <summary>
        /// Normalise the values of a data point according to any fitted parameters.
        /// </summary>
        /// <param name="dataPoint"></param>
        /// <returns></returns>
        void Normalise(DataPoint dataPoint);

        /// <summary>
        /// Denormalise the values of a data point according to any fitted parameters.
        /// </summary>
        /// <param name="dataPoint"></param>
        /// <returns></returns>
        void Denormalise(DataPoint dataPoint);

    }

}
