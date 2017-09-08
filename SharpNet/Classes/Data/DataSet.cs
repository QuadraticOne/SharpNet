using SharpNet.Classes.Utils;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.Data
{

    /// <summary>
    /// Abstract data set class from which other kinds of data sets are derived.
    /// </summary>
    public abstract class DataSet
    {

        public enum Type { REGRESSION, CLASSIFICATION, UNSUPERVISED }
        public Type DataSetType { get; private set; }

        public DataPoint[] TrainingSet { get; private set; }
        public DataPoint[] ValidationSet { get; private set; }
        public DataPoint[] TestSet { get; private set; }

        public int Inputs { get; private set; }
        public int Outputs { get; private set; }
        
        private Queue<DataPoint> unassigned = new Queue<DataPoint>();
        private Random random = new Random();

        /// <summary>
        /// Base constructor; sets up set arrays.
        /// </summary>
        public DataSet()
        {
            TrainingSet = new DataPoint[0];
            ValidationSet = new DataPoint[0];
            TestSet = new DataPoint[0];
        }

        /// <summary>
        /// Add a pre-existing data point to the data set.
        /// </summary>
        /// <param name="dataPoint"></param>
        public void AddDataPoint(DataPoint dataPoint)
        {
            unassigned.Enqueue(dataPoint);
        }

        /// <summary>
        /// Randomly assign any unassigned points to the training, validation, and test sets with
        /// the given ratio of probabilities.
        /// </summary>
        /// <param name="trainingRatio"></param>
        /// <param name="validationRatio"></param>
        /// <param name="testRatio"></param>
        public void AssignDataPoints(double trainingRatio, double validationRatio,
            double testRatio)
        {
            // Calculate probability cutoffs for assignment
            double cutoff1 = trainingRatio / (trainingRatio + validationRatio + testRatio);
            double cutoff2 = (trainingRatio + validationRatio) /
                (trainingRatio + validationRatio + testRatio);

            List<DataPoint> training = new List<DataPoint>();
            List<DataPoint> validation = new List<DataPoint>();
            List<DataPoint> test = new List<DataPoint>();

            while (unassigned.Count > 0)
            {
                double randomDouble = random.NextDouble();
                if (randomDouble < cutoff1) training.Add(unassigned.Dequeue());
                else if (randomDouble < cutoff2) validation.Add(unassigned.Dequeue());
                else test.Add(unassigned.Dequeue());
            }

            // Construct new arrays
            DataPoint[] trainingBuffer = new DataPoint[TrainingSet.Length + training.Count];
            TrainingSet.CopyTo(trainingBuffer, training.Count);
            training.CopyTo(trainingBuffer, 0);
            TrainingSet = trainingBuffer;

            DataPoint[] validationBuffer = new DataPoint[ValidationSet.Length + validation.Count];
            ValidationSet.CopyTo(validationBuffer, validation.Count);
            validation.CopyTo(validationBuffer, 0);
            ValidationSet = validationBuffer;

            DataPoint[] testBuffer = new DataPoint[TestSet.Length + test.Count];
            TestSet.CopyTo(testBuffer, test.Count);
            test.CopyTo(testBuffer, 0);
            TestSet = testBuffer;
        }

        /// <summary>
        /// Returns a collation of all unassigned, training, validation, and test data points.
        /// </summary>
        /// <returns></returns>
        public DataPoint[] GetWholeSet()
        {
            DataPoint[] set = new DataPoint[unassigned.Count + TrainingSet.Length +
                ValidationSet.Length + TestSet.Length];

            TrainingSet.CopyTo(set, 0);
            ValidationSet.CopyTo(set, TrainingSet.Length);
            TestSet.CopyTo(set, TrainingSet.Length + ValidationSet.Length);
            unassigned.CopyTo(set, set.Length - unassigned.Count);

            return set;
        }

        /// <summary>
        /// Return a training example by index.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public DataPoint GetTrainingExample(int index)
        {
            return TrainingSet[index];
        }

        /// <summary>
        /// Return a random training example.
        /// </summary>
        /// <returns></returns>
        public DataPoint GetRandomTrainingExample()
        {
            return TrainingSet[(int) Math.Floor(random.NextDouble() * TrainingSet.Length)];
        }

        /// <summary>
        /// Returns a random subset of the training set.
        /// </summary>
        /// <param name="count"></param>
        /// <returns></returns>
        public DataPoint[] GetRandomTrainingSubset(int count)
        {
            DataPoint[] subset = new DataPoint[count];
            for (int i = 0; i < subset.Length; i++) subset[i] = GetRandomTrainingExample();
            return subset;
        }

        /// <summary>
        /// Return a validation example by index.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public DataPoint GetValidationExample(int index)
        {
            return ValidationSet[index];
        }

        /// <summary>
        /// Return a random validation example.
        /// </summary>
        /// <returns></returns>
        public DataPoint GetRandomValidationExample()
        {
            return ValidationSet[(int)Math.Floor(random.NextDouble() * ValidationSet.Length)];
        }

        /// <summary>
        /// Returns a random subset of the validation set.
        /// </summary>
        /// <param name="count"></param>
        /// <returns></returns>
        public DataPoint[] GetRandomValidationSubset(int count)
        {
            DataPoint[] subset = new DataPoint[count];
            for (int i = 0; i < subset.Length; i++) subset[i] = GetRandomValidationExample();
            return subset;
        }

        /// <summary>
        /// Return a test example by index.
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public DataPoint GetTestExample(int index)
        {
            return TrainingSet[index];
        }

        /// <summary>
        /// Return a random test example.
        /// </summary>
        /// <returns></returns>
        public DataPoint GetRandomTestExample()
        {
            return TestSet[(int)Math.Floor(random.NextDouble() * TestSet.Length)];
        }

        /// <summary>
        /// Returns a random subset of the test set.
        /// </summary>
        /// <param name="count"></param>
        /// <returns></returns>
        public DataPoint[] GetRandomTestSubset(int count)
        {
            DataPoint[] subset = new DataPoint[count];
            for (int i = 0; i < subset.Length; i++) subset[i] = GetRandomTestExample();
            return subset;
        }

        /// <summary>
        /// Normalise the data points in this set according to a data normaliser.  If the data
        /// normaliser has not been fit to this set, that will be done automatically.
        /// </summary>
        /// <param name="dataNormaliser"></param>
        public void NormaliseBy(IDataNormaliser dataNormaliser)
        {
            if (!dataNormaliser.HasBeenFit()) dataNormaliser.Fit(this.GetWholeSet());

            foreach (DataPoint dataPoint in GetWholeSet()) dataNormaliser.Normalise(dataPoint);
        }

        /// <summary>
        /// Denormalise the data points in this set according to a data normaliser.  If the data
        /// normaliser has not been fit, this will throw an error; it makes no sense to fit a model
        /// to data, then immediately denormalise them according to it.
        /// </summary>
        /// <param name="dataNormaliser"></param>
        public void DenormaliseBy(IDataNormaliser dataNormaliser)
        {
            if (!dataNormaliser.HasBeenFit()) throw new OperationCanceledException(
                "Cannot denormalise data according to a normaliser that has not been fit.");

            foreach (DataPoint dataPoint in GetWholeSet()) dataNormaliser.Denormalise(dataPoint);
        }

        /// <summary>
        /// Data set for storing and accessing regression training examples.
        /// </summary>
        public class Regression : DataSet
        {

            /// <summary>
            /// Create a new regression data set.
            /// </summary>
            /// <param name="inputs"></param>
            /// <param name="outputs"></param>
            public Regression(int inputs, int outputs) : base()
            {
                DataSetType = Type.REGRESSION;
                Inputs = inputs;
                Outputs = outputs;
            }

            /// <summary>
            /// Add a new labelled regression training example.
            /// </summary>
            /// <param name="input"></param>
            /// <param name="output"></param>
            public void AddDataPoint(double[] input, double[] output)
            {
                unassigned.Enqueue(new DataPoint(input, output));
            }

        }

        /// <summary>
        /// Data set for storing and accessing classification training examples.
        /// </summary>
        public class Classification : DataSet
        {

            /// <summary>
            /// Create a new classification data set.
            /// </summary>
            /// <param name="inputs"></param>
            public Classification(int inputs) : base()
            {
                DataSetType = Type.CLASSIFICATION;
                Inputs = inputs;
                Outputs = 1;
            }

            /// <summary>
            /// Add a new labelled classification training example.
            /// </summary>
            /// <param name="input"></param>
            /// <param name="category"></param>
            public void AddDataPoint(double[] input, int category)
            {
                unassigned.Enqueue(new DataPoint(input, category));
            }

        }

        /// <summary>
        /// Data set for storing and accessing unsupervised training examples.
        /// </summary>
        public class Unsupervised : DataSet
        {

            /// <summary>
            /// Create a new unsupervised data set.
            /// </summary>
            /// <param name="inputs"></param>
            public Unsupervised(int inputs) : base()
            {
                DataSetType = Type.UNSUPERVISED;
                Inputs = inputs;
                Outputs = 0;
            }

            /// <summary>
            /// Add a new unlabelled training example.
            /// </summary>
            /// <param name="input"></param>
            public void AddDataPoint(double[] input)
            {
                unassigned.Enqueue(new DataPoint(input));
            }

        }

    }

}
