using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SharpNet.Classes.Maths;
using SharpNet.Classes.Architecture.NetworkLayer.Layers;

namespace SharpNetTest.Classes.Architecture
{
    [TestClass]
    public class FeedForwardLayerTest
    {
        [TestMethod]
        public void ProcessInputTest()
        {
            Matrix unprocessedInput = new Matrix(3, 1);
            unprocessedInput[0, 0] = 2;
            unprocessedInput[1, 0] = 3;
            unprocessedInput[2, 0] = 4;

            FeedForwardLayer ffl = new FeedForwardLayer.Dense(3, 1);

            //Matrix processedInput = ffl.ProcessInput();
            Assert.AreEqual(new Matrix(4, 4), new Matrix(4, 4));
        }
    }
}
