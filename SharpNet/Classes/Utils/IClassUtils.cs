using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Utils
{

    /// <summary>
    /// Implementing classes are required to define a number of quality-of-life methods, making
    /// tasks such as loading and saving objects, as well as debugging, easier.  When implementing
    /// the interface, the generic parameter T should be set to the implementing class.
    /// </summary>
    interface IClassUtils<T>
    {

        /// <summary>
        /// Represent the important contents of the object in a human-legible string format, which
        /// could be printed to the console.
        /// </summary>
        /// <returns></returns>
        string ToString();

        /// <summary>
        /// Make a deep copy of the object.
        /// </summary>
        /// <returns></returns>
        T Copy();

        /// <summary>
        /// Return a deterministic representation of the object in the form of an array of doubles.
        /// </summary>
        /// <returns></returns>
        double[] ToArray();

        /// <summary>
        /// From an empty object, construct a copy of the original object on which ToArray() was
        /// called to produce the input array.
        /// </summary>
        /// <param name="array"></param>
        void FromArray(double[] array);

    }

}
