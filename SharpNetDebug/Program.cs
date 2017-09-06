using System;
using SharpNet.Classes.Maths;

namespace SharpNetDebug
{

    /// <summary>
    /// Program for debugging SharpNet.  Does not contain any useful code.
    /// </summary>
    public class Program
    {

        /// <summary>
        /// Debugs SharpNet.  Code added here will be run when the project is run.
        /// </summary>
        /// <param name="args"></param>
        static void Main(string[] args)
        {

            Console.WriteLine("Debugging SharpNet.");

            #region DEBUG_CODE
            Vector v = new Vector(4);
            v.RandomUniform(-1, 1);
            Console.WriteLine(v.ToDetailedString());
            Vector w = new Vector(4);
            w.RandomUniform(-1, 1);
            Console.WriteLine(w.ToDetailedString());
            Console.WriteLine(v.Dot(w));
            #endregion

            Console.WriteLine("Debugging has finished.  Press ENTER to exit.");
            Console.Read();

        }

    }

}
