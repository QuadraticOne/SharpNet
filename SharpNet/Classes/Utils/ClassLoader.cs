using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Utils
{

    /// <summary>
    /// Loads and saves classes.
    /// </summary>
    public static class ClassLoader
    {

        /// <summary>
        /// Take the string representation of an array of doubles which contains information about
        /// an instance of a class, and from that built the instance.  Then return the instance.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="inputString"></param>
        /// <returns></returns>
        private static T StringToObject<T>(string inputString) where T : IClassUtils<T>, new()
        {
            T newObject = new T();

            string[] stringArray = inputString.Split(new char[] { ',' });
            double[] doubleArray = new double[stringArray.Length];

            newObject.FromArray(doubleArray);
            return newObject;
        }

    }

}
