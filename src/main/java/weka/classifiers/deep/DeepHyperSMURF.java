package weka.classifiers.deep;

import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import weka.classifiers.Classifier;
import weka.classifiers.trees.HyperSMURF;
import weka.core.Instance;
import weka.core.Instances;

public class DeepHyperSMURF extends HyperSMURF {

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
		System.out.println("Start collecting data.");
		double[][] doubleData = new double[d.numInstances()][m_Classifiers.length];
		double[] doubleLabelsPos = new double[d.numInstances()];
		double[] doubleLabelsNeg = new double[d.numInstances()];

		int i = 0;
		for (Instance instance : d) {
			doubleLabelsPos[i] = instance.classValue();
			doubleLabelsNeg[i] = (instance.classValue() == 1 ? 0 : 1);
			int j = 0;
			for (Classifier classifier : m_Classifiers) {
				try {
					double[] prediction = classifier.distributionForInstance(instance);
					doubleData[i][j] = prediction[1];
				} catch (Exception e) {
					e.printStackTrace();
					throw new RuntimeException(e);
				}
				j++;
			}

			i++;
		}
		INDArray data = Nd4j.create(doubleData);
		INDArray labels = Nd4j.hstack(Nd4j.create(doubleLabelsNeg, new int[] { m_Classifiers.length, 1 }));//,
			//	Nd4j.create(doubleLabelsPos, new int[] { m_Classifiers.length, 1 }));
		// save memory;
		doubleData = null;

		int batchSize = 1000;

		m_deepData = new DataSet(data, labels);
		List<DataSet> listDs = m_deepData.asList();
		Collections.shuffle(listDs, new Random(getSeed()));
		ListDataSetIterator dataIterator = new ListDataSetIterator(listDs, batchSize);

		int numInput = m_Classifiers.length;
		int numOutputs = 1;
		int nHidden = 1;
		int listenerFreq = batchSize / 5;

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder().seed(getSeed()).activation("relu")
				.iterations(1).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).updater(Updater.ADAGRAD)
				.weightInit(WeightInit.XAVIER).weightInit(WeightInit.SIZE).learningRate(0.2).regularization(true).l2(0.0001).list()
				.layer(0,
						new DenseLayer.Builder()
								// create the first, input layer with xavier initialization
								.nIn(numInput).nOut(nHidden).weightInit(WeightInit.XAVIER).build())
				.layer(1,
						new OutputLayer.Builder().nIn(nHidden).nOut(numOutputs).weightInit(WeightInit.XAVIER)
								.lossFunction(LossFunctions.LossFunction.MSE).build())
				.pretrain(false).backprop(true) // use backpropagation to adjust weights
				.build();

		m_deepClassifier = new MultiLayerNetwork(conf);
		m_deepClassifier.init();
		// print the score with every 1 iteration
		m_deepClassifier
				.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

		try {
			while (dataIterator.hasNext()) {
				DataSet ds = dataIterator.next();
				m_deepClassifier.fit(ds.getFeatures(), ds.getLabels());
				// m_deepClassifier.fit(m_deepData);
			}
		} catch (Exception e) {
			e.printStackTrace();

		}

		System.out.println("Ready");

	}

	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		INDArray dataInstance = Nd4j.zeros(getNumIterations());
		int i = 0;
		for (Classifier classifier : m_Classifiers) {
			try {
				dataInstance.putScalar(i, classifier.classifyInstance(instance));
			} catch (Exception e) {
				e.printStackTrace();
				throw new RuntimeException();
			}
			i++;
		}
		try {
			double[][] a = new double[2][1];
			a[0][0] = 0.0;
			a[1][0] = 1.0;
			double[][] b = new double[2][1];
			a[0][0] = 1.0;
			a[1][0] = 0.0;
			INDArray l1 = Nd4j.create(a);
			Evaluation eval = new Evaluation();
			INDArray l2 = Nd4j.create(b);
			INDArray test = m_deepClassifier.output(dataInstance, false);
			
//			int[] k = m_deepClassifier.predict(dataInstance);
			return new double[] {test.getDouble(0),1-test.getDouble(0)};
		} catch (Exception e) {
			e.printStackTrace();
			throw new RuntimeException();
		}
	}

}
