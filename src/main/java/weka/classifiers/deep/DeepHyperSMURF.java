package weka.classifiers.deep;

import java.util.List;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import weka.classifiers.Classifier;
import weka.classifiers.trees.HyperSMURF;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SpreadSubsample;

public class DeepHyperSMURF extends HyperSMURF {

	private static Logger log = LoggerFactory.getLogger(DeepHyperSMURF.class);

	/**
	 * Default serial ID for serialization
	 */
	private static final long serialVersionUID = -4302242342854101700L;

	private MultiLayerNetwork m_deepClassifier;

	private DataSet m_deepData;

	@Override
	public void buildClassifier(Instances data) throws Exception {

		super.buildClassifier(data);

		this.buildDeepHyperSMURF(data);
	}

	private void buildDeepHyperSMURF(Instances d) {
		// balance the data first!
		SpreadSubsample resample = new SpreadSubsample();
		resample.setRandomSeed(getSeed());
		resample.setDistributionSpread(1.0);

		try {
			resample.setInputFormat(d);
			d = Filter.useFilter(d, resample);
			
			log.info("Build data....");
			double[][] doubleData = new double[d.numInstances()][m_Classifiers.length];
			double[][] doubleLabel = new double[d.numInstances()][d.numClasses()];

			int i = 0;
			for (Instance instance : d) {
				doubleLabel[i][(int) instance.classValue()] = 1.0;

				Instance classMissing = (Instance) instance.copy();
			    classMissing.setDataset(instance.dataset());
			    classMissing.setClassMissing();
				
				int j = 0;
				for (Classifier classifier : m_Classifiers) {
					try {
						double[] prediction = classifier.distributionForInstance(classMissing);
						doubleData[i][j] = prediction[0];
					} catch (Exception e) {
						e.printStackTrace();
						throw new RuntimeException(e);
					}
					j++;
				}

				i++;
			}
			INDArray data = Nd4j.create(doubleData);
			INDArray labels = Nd4j.create(doubleLabel);// ,
			// Nd4j.create(doubleLabelsPos, new int[] { m_Classifiers.length, 1 }));
			// save memory;
			doubleData = null;

			int batchSize = 10000;

			m_deepData = new DataSet(data, labels);
			List<DataSet> listDs = m_deepData.asList();
			ListDataSetIterator dataIterator = new ListDataSetIterator(listDs, batchSize);

			log.info("Build model....");

			int numInput = m_Classifiers.length;
			int numOutputs = 2;
			int numEpochs = 1000;
			int nHidden = 1;
			int listenerFreq = numEpochs / 5;

			MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(getSeed()).iterations(numEpochs)
					.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).activation("tanh")
					.weightInit(WeightInit.XAVIER).learningRate(0.1)
					.regularization(true).l2(1e-4).list()
					.layer(0, new DenseLayer.Builder()
							// create the first, input layer with xavier initialization
							.nIn(numInput).nOut(nHidden).build())
					.layer(1,
							new OutputLayer.Builder().nIn(nHidden).nOut(numOutputs)
									.activation("softmax")
									.lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).build())
					.pretrain(false).backprop(true) // use backpropagation to adjust weights
					.build();

			m_deepClassifier = new MultiLayerNetwork(conf);
			m_deepClassifier.init();
			// print the score with every 1 iteration
			m_deepClassifier.setListeners(new ScoreIterationListener(listenerFreq));

			log.info("Train model....");
			m_deepClassifier.fit(dataIterator);

			log.info("Ready");

		} catch (Exception e) {
			e.printStackTrace();

		}

	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		try {
			double[] doubleInstance = new double[m_Classifiers.length];
			int i = 0;
			for (Classifier classifier : m_Classifiers) {
				try {
					doubleInstance[i] = classifier.distributionForInstance(instance)[0];
				} catch (Exception e) {
					e.printStackTrace();
					throw new RuntimeException();
				}
				i++;
			}
			INDArray dataInstance = Nd4j.create(doubleInstance);

			INDArray test = m_deepClassifier.output(dataInstance, false);

			// int[] k = m_deepClassifier.predict(dataInstance);
			return new double[] { test.getDouble(0), test.getDouble(1) };
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException();
		}
	}

}
