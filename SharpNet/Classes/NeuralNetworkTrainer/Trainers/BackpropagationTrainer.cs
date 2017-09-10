using SharpNet.Classes.Architecture.NetworkLayer.Layers;
using SharpNet.Classes.Data;
using SharpNet.Classes.Maths;
using SharpNet.Classes.Maths.Error;
using SharpNet.Classes.NeuralNetwork.NeuralNetworks;
using System;
using System.Collections.Generic;
using System.Text;

namespace SharpNet.Classes.NeuralNetworkTrainer
{

    /// <summary>
    /// Class which trains feedforward networks on data for regression or classification problems.
    /// </summary>
    public class BackpropagationTrainer : ITrainer
    {

        // MARK: required elements

        private bool _individualLearningRates = false;
        public bool IndividualLearningRates
        {
            get { return _individualLearningRates; }
            
            set
            {
                _individualLearningRates = value;
                TrainingSetup();
            }
        }

        private double _learningRate = 0;
        public double LearningRate
        {
            get
            {
                if (IndividualLearningRates) throw new MemberAccessException(
                    "Cannot access LearningRate while IndividualLearningRates is true.  Use " +
                    "the GetLearningRate method instead.");
                return _learningRate;
            }

            set
            {
                SetLearningRate(value);
            }
        }

        public IInitialiser initialiser = null;
        public ILossFunction lossFunction = null;

        public List<ITerminationCondition> terminationConditions =
            new List<ITerminationCondition>();

        public delegate DataPoint[] BatchSelector(DataSet dataSet);
        public BatchSelector batchSelector = null;

        private Vector[] vectorLearningRates = null;
        private Matrix[] matrixLearningRates = null;

        private FeedForwardNetwork Network = null;
        private DataSet DataSet = null;

        private int epochCount = 0;
        private int iterationCount = 0;
        private int exampleCount = 0;

        private FeedForwardLayer.Gradient[] gradients;

        public bool IsTraining { get; private set; } = false;

        // MARK: optional elements

        public List<IRegulariser> regularisers = new List<IRegulariser>();
        public List<ITrainingModifier> modifiers = new List<ITrainingModifier>();

        /// <summary>
        /// Whether or not to update network weights after each training example.
        /// </summary>
        public bool stochastic = false;

        /// <summary>
        /// After how many epochs the network should be evaluated.
        /// </summary>
        public int evaluationFrequency = 10;

        /// <summary>
        /// Whether or not to log data.
        /// </summary>
        public bool logging = false;

        public List<double[]> evaluations = new List<double[]>();

        // MARK: public methods

        /// <summary>
        /// Use this backpropagation trainer to 
        /// </summary>
        /// <param name="network"></param>
        /// <param name="dataSet"></param>
        public void Train(FeedForwardNetwork network, DataSet dataSet)
        {
            Network = network;
            DataSet = dataSet;
            TrainingSetup();
            if (!IsReady()) throw new OperationCanceledException(
                "The trainer does not yet have the required information to start training.");

            StartTraining();
        }

        /// <summary>
        /// Return the learning rate.  Throws an error if IndividualLearningRates is set to true.
        /// </summary>
        /// <returns></returns>
        public double GetLearningRate()
        {
            if (IndividualLearningRates) throw new MemberAccessException(
                "Cannot access LearningRate while IndividualLearningRates is true.  Use " +
                "the GetLearningRate method instead.");
            return _learningRate;
        }

        /// <summary>
        /// Return the learning rate of a specific neuron in a layer, assuming that the layer is a
        /// sparse layer.
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="index"></param>
        /// <returns></returns>
        public double GetLearningRate(int layer, int index)
        {
            if (!IndividualLearningRates) throw new MemberAccessException(
                "This overload can only be used when IndividualLearningRates is set to true.");
            if (!LayerIsSparse(layer)) throw new MemberAccessException(
                "This overload should only be called on sparse layers.");
            return vectorLearningRates[layer][index];
        }

        /// <summary>
        /// Return the learning rate relating to the weight between two specific neurons in a
        /// layer.  This assumes both that the layer is a dense layer, and that
        /// IndividualLearningRates is set to true.
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="inputIndex"></param>
        /// <param name="outputIndex"></param>
        /// <returns></returns>
        public double GetLearningRate(int layer, int inputIndex, int outputIndex)
        {
            if (!IndividualLearningRates) throw new MemberAccessException(
                "This overload can only be used when IndividualLearningRates is set to true.");
            if (!LayerIsDense(layer)) throw new MemberAccessException(
                "This overload should only be called on dense layers.");
            return matrixLearningRates[layer][outputIndex, inputIndex];
        }

        /// <summary>
        /// Set the learning rate of the network.  If individual learning rates are being used, all
        /// learning rates will be set to this value.
        /// </summary>
        /// <param name="learningRate"></param>
        public void SetLearningRate(double learningRate)
        {
            _learningRate = learningRate;

            // If network has not been given yet, save learning rate to single value then update
            // later
            if (IndividualLearningRates && (Network != null))
            {
                for (int i = 0; i < Network.Layers.Count; i++)
                {
                    if (vectorLearningRates[i] != null)
                    {
                        for (int j = 0; j < vectorLearningRates[i].Length; i++)
                            vectorLearningRates[i][j] = LearningRate;
                    }

                    if (matrixLearningRates[i] != null)
                    {
                        for (int j = 0; j < matrixLearningRates[i].Rows; j++)
                        {
                            for (int k = 0; k < matrixLearningRates[i].Columns; k++)
                                matrixLearningRates[i][j, k] = LearningRate;
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Set the learning rate of a specific neuron in a layer, assuming that layer is a sparse
        /// layer.
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="index"></param>
        /// <param name="learningRate"></param>
        public void SetLearningRate(int layer, int index, double learningRate)
        {
            if (!IndividualLearningRates) throw new MemberAccessException(
                "This overload can only be used when IndividualLearningRates is set to true.");
            if (!LayerIsSparse(layer)) throw new MemberAccessException(
                "This overload should only be called on sparse layers.");
            vectorLearningRates[layer][index] = learningRate;
        }

        /// <summary>
        /// Set the learning rate of a specific neuron in a layer, assuming that layer is a dense
        /// one.
        /// </summary>
        /// <param name="layer"></param>
        /// <param name="inputIndex"></param>
        /// <param name="outputIndex"></param>
        /// <param name="learninRate"></param>
        public void SetLearningRate(int layer, int inputIndex, int outputIndex, double learninRate)
        {
            if (!IndividualLearningRates) throw new MemberAccessException(
                "This overload can only be used when IndividualLearningRates is set to true.");
            if (!LayerIsDense(layer)) throw new MemberAccessException(
                "This overload should only be called on dense layers.");
            matrixLearningRates[layer][outputIndex, inputIndex] = learninRate;
        }

        /// <summary>
        /// Returns true if the layer at an index is dense; false otherwise.  Throws an error if a
        /// network has not been given to the trainer.
        /// </summary>
        /// <param name="layer"></param>
        /// <returns></returns>
        public bool LayerIsDense(int layer)
        {
            if (Network == null) throw new OperationCanceledException(
                "The trainer has not been given a network.");

            return (Network.Layers[layer] is FeedForwardLayer.Dense);
        }

        /// <summary>
        /// Returns true if the layer at an index is sparse; false otherwise.  Throws an error if a
        /// network has not been given to the trainer.
        /// </summary>
        /// <param name="layer"></param>
        /// <returns></returns>
        public bool LayerIsSparse(int layer)
        {
            if (Network == null) throw new OperationCanceledException(
                "The trainer has not been given a network.");

            return (Network.Layers[layer] is FeedForwardLayer.Sparse);
        }

        /// <summary>
        /// Return the current epoch of the trainer.
        /// </summary>
        /// <returns></returns>
        public int GetEpoch()
        {
            return epochCount;
        }

        /// <summary>
        /// Calculate and return the loss of the network on a specific data point.
        /// </summary>
        /// <param name="example"></param>
        /// <returns></returns>
        public double Loss(DataPoint example)
        {
            Network.Input = Matrix.ToColumnMatrix(example.input);
            double loss = lossFunction.Error(Network.Output, Matrix.ToColumnMatrix(example.output));
            foreach (IRegulariser regulariser in regularisers)
                loss += regulariser.Loss(Network);
            return loss;
        }

        /// <summary>
        /// Evaluate the training loss of the network.
        /// </summary>
        /// <returns></returns>
        public double TrainingLoss()
        {
            double error = 0;
            foreach (DataPoint dataPoint in DataSet.TrainingSet)
                error += Loss(dataPoint);
            return error / DataSet.TrainingSet.Length;
        }

        /// <summary>
        /// Evaluate the validation loss of the network.
        /// </summary>
        /// <returns></returns>
        public double ValidationLoss()
        {
            double error = 0;
            foreach (DataPoint dataPoint in DataSet.ValidationSet)
                error += Loss(dataPoint);
            return error / DataSet.ValidationSet.Length;
        }

        /// <summary>
        /// Evaluate the network.  Output is an array of three doubles: epoch count, training 
        /// error, and validation error.
        /// </summary>
        /// <returns></returns>
        public double[] EvaluateNetwork()
        {
            double[] evaluation = new double[3];
            evaluation[0] = epochCount;
            evaluation[1] = TrainingLoss();
            evaluation[2] = ValidationLoss();
            return evaluation;
        }

        /// <summary>
        /// Returns true if the network is ready.
        /// </summary>
        /// <returns></returns>
        public bool IsReady()
        {
            bool ready = true;

            if (Network == null) ready = false;
            if (initialiser == null) ready = false;
            if (lossFunction == null) ready = false;
            if (IndividualLearningRates)
            {
                if ((vectorLearningRates == null) || (matrixLearningRates == null)) ready = false;
            }
            else
            {
                if (_learningRate == 0) ready = false;
            }
            if (terminationConditions.Count < 1) ready = false;
            if (batchSelector == null) ready = false;

            return ready;
        }

        /// <summary>
        /// Return an array of strings, which lists any problems preventing the trainer from starting.
        /// </summary>
        /// <returns></returns>
        public string[] Troubleshoot()
        {
            List<string> list = new List<string>();

            if (Network == null) list.Add("no network");
            if (initialiser == null) list.Add("no initialiser");
            if (lossFunction == null) list.Add("no loss function");
            if (IndividualLearningRates)
            {
                if ((vectorLearningRates == null) || (matrixLearningRates == null)) list.Add(
                    "individual learning rates is true, but no learning rate lists");
            }
            else
            {
                if (_learningRate <= 0) list.Add("learning rate is 0 or negative");
            }
            if (terminationConditions.Count < 1) list.Add("no termination conditions");
            if (batchSelector == null) list.Add("no batch selector");

            if (list.Count == 0) list.Add("no visible issues");

            string[] array = new string[list.Count];
            list.CopyTo(array);
            return array;
        }

        // MARK: private methods

        /// <summary>
        /// Performs any setup necessary for training.
        /// </summary>
        private void TrainingSetup()
        {
            if (Network != null)
            {
                if (IndividualLearningRates)
                {
                    vectorLearningRates = new Vector[Network.Layers.Count];
                    matrixLearningRates = new Matrix[Network.Layers.Count];

                    for (int i = 0; i < Network.Layers.Count; i++)
                    {
                        if (LayerIsDense(i))
                            matrixLearningRates[i] = new Matrix(Network.Layers[i].Outputs,
                                Network.Layers[i].Inputs + 1);
                        else matrixLearningRates[i] = null;

                        if (LayerIsSparse(i))
                            throw new NotImplementedException("Sparse layers are NYI.");
                        else vectorLearningRates[i] = null;
                    }
                }

                SetLearningRate(_learningRate);
            }
        }

        /// <summary>
        /// Return true if one or more of the termination conditions for this trainer have been
        /// reached.
        /// </summary>
        /// <returns></returns>
        private bool IsFinished()
        {
            foreach (ITerminationCondition condition in terminationConditions)
                if (condition.HasFinished(this)) return true;
            return false;
        }

        /// <summary>
        /// Begin training, and continue until one or more termination conditions are reached.
        /// </summary>
        private void StartTraining()
        {
            IsTraining = true;
            epochCount = 0;
            evaluations.Clear();

            initialiser.Initialise(Network);

            gradients = new FeedForwardLayer.Gradient[Network.Layers.Count];
            for (int i = 0; i < Network.Layers.Count; i++)
            {
                if (LayerIsDense(i)) gradients[i] = new FeedForwardLayer.Dense.Gradient(
                    (FeedForwardLayer.Dense)Network.Layers[i]);  // Can safely be cast
                if (LayerIsSparse(i)) gradients[i] = new FeedForwardLayer.Sparse.Gradient(
                    (FeedForwardLayer.Sparse)Network.Layers[i]);  // Can safely be cast
            }

            while (!IsFinished())
            {
                if (epochCount % evaluationFrequency == 0) evaluations.Add(EvaluateNetwork());
                PerformEpoch();
            }

            IsTraining = false;
        }

        /// <summary>
        /// Perform an epoch on the training data.
        /// </summary>
        private void PerformEpoch()
        {
            epochCount++;
            iterationCount = 0;
            exampleCount = 0;
            while (exampleCount < DataSet.TrainingSet.Length) PerformIteration();
        }

        /// <summary>
        /// Perform one iteration on the training data.
        /// </summary>
        private void PerformIteration()
        {
            iterationCount++;
            DataPoint[] batch = batchSelector(DataSet);
            exampleCount += batch.Length;

            foreach (DataPoint dataPoint in batch)
            {
                Network.Input = Matrix.ToColumnMatrix(dataPoint.input);
                gradients[gradients.Length - 1].Backpropagate(
                    Matrix.ToColumnMatrix(dataPoint.output), lossFunction, regularisers);

                for (int i = gradients.Length - 2; i >= 0; i--)
                {
                    gradients[i].Backpropagate(gradients[i + 1], regularisers);
                }

                if (stochastic) ApplyDeltas();
            }

            ApplyDeltas();
        }

        /// <summary>
        /// Apply all accumulated deltas to their targets.
        /// </summary>
        private void ApplyDeltas()
        {
            int layerIndex = 0;
            foreach (FeedForwardLayer.Gradient gradient in gradients)
            {
                layerIndex++;
                if (!IndividualLearningRates) gradient.ApplyDeltas(LearningRate);
                else
                {
                    if (gradient is FeedForwardLayer.Dense.Gradient)
                        gradient.ApplyDeltas(matrixLearningRates[layerIndex]);
                    if (gradient is FeedForwardLayer.Sparse.Gradient)
                        gradient.ApplyDeltas(vectorLearningRates[layerIndex]);
                }
            }
        }

    }

}
