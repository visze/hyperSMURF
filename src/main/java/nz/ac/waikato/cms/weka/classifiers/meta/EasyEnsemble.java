package nz.ac.waikato.cms.weka.classifiers.meta;

import java.util.Random;

import weka.classifiers.RandomizableParallelIteratedSingleClassifierEnhancer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveWithValues;

/**
 * <!-- globalinfo-start -->
 * 
 * Class for EasyEnsemble. EasyEnsemble splits up the majority class into n partitions and trains n classifiers using
 * the majority partition and all minority classes.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Liu X, Wu J, Zhou Z. Exploratory undersampling for class-imbalance learning. Systems,Man, and Cybernetics, Part B:
 * Cybernetics, IEEE Transactions on. 2009;39(2):539–550.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;article{Liu09,
 *    Author = {Liu, X. and Wu, J. and Zhou, Z.},
 *    Journal = {Systems, Man, and Cybernetics, Part B: Cybernetics, IEEE Transactions on},
 *    Month = {April},
 *    Number = {2},
 *    Pages = {539-550},
 *    Title = {Exploratory Undersampling for Class-Imbalance Learning},
 *    Volume = {39},
 *    Year = {2009}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start --> Valid options are:
 * <p/>
 * 
 * <pre>
 *  -P &lt;num&gt;
 *  Number of partitions of the majority class. 
 *  (default 10)
 * </pre>
 * 
 * 
 * <pre>
 *  -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)
 * </pre>
 * 
 * <pre>
 *  -num-slots &lt;num&gt;
 *  Number of execution slots.
 *  (default 1 - i.e. no parallelism)
 * </pre>
 * 
 * 
 * <pre>
 *  -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <pre>
 *  -W
 *  Full name of base classifier.
 *  (default: weka.classifiers.rules.ZeroR)
 * </pre>
 * 
 * <pre>
 * 
 * <!-- options-end -->
 *
 * Options after -- are passed to the designated classifier.
 * <p>
 *
 * @author <a href="mailto:max.schubach@charite.de">Max Schubach</a>
 * @version 0.1
 * 
 *
 */
public class EasyEnsemble extends RandomizableParallelIteratedSingleClassifierEnhancer {

	protected Instances m_majorityData;
	protected Instances m_minorityData;
	protected Instances m_data;
	protected Random m_random;

	/** for serialization */
	private static final long serialVersionUID = 3340927280517126814L;

	/**
	 * Constructor.
	 */
	public EasyEnsemble() {
		super();
	}

	/**
	 * EasyEnsemble method.
	 *
	 * @param data
	 *            the training data to be used for generating the EasyEnsemble classifier.
	 * @throws Exception
	 *             if the classifier could not be built successfully
	 */
	@Override
	public void buildClassifier(Instances data) throws Exception {

		getCapabilities().testWithFail(data);

		// remove instances with missing class
		m_data = new Instances(data);
		m_data.deleteWithMissingClass();

		super.buildClassifier(m_data);

		RemoveWithValues classValueFilter = new RemoveWithValues();
		classValueFilter.setAttributeIndex(Integer.toString(data.classIndex() + 1));
		classValueFilter.setNominalIndicesArr(new int[] { getMinorityClass(data) + 1 });
		classValueFilter.setInputFormat(m_data);

		m_minorityData = Filter.useFilter(m_data, classValueFilter);

		classValueFilter.setInvertSelection(true);
		classValueFilter.setInputFormat(m_data);
		m_majorityData = Filter.useFilter(m_data, classValueFilter);

		// save memory
		m_data = null;

		m_random = new Random(m_Seed);

		for (int j = 0; j < m_Classifiers.length; j++) {
			if (m_Classifier instanceof Randomizable) {
				((Randomizable) m_Classifiers[j]).setSeed(m_random.nextInt());
			}
		}

		buildClassifiers();

		// save memory
		m_majorityData = null;
		m_minorityData = null;

	}

	/**
	 * Calculates the class membership probabilities for the given test instance.
	 *
	 * @param instance
	 *            the instance to be classified
	 * @return predicted class probability distribution
	 * @throws Exception
	 *             if distribution can't be computed successfully
	 */
	@Override
	public double[] distributionForInstance(Instance instance) throws Exception {
		double[] sums = new double[instance.numClasses()], newProbs;

		double numPreds = 0;
		for (int i = 0; i < m_NumIterations; i++) {
			if (instance.classAttribute().isNumeric() == true) {
				double pred = m_Classifiers[i].classifyInstance(instance);
				if (!Utils.isMissingValue(pred)) {
					sums[0] += pred;
					numPreds++;
				}
			} else {
				newProbs = m_Classifiers[i].distributionForInstance(instance);
				for (int j = 0; j < newProbs.length; j++)
					sums[j] += newProbs[j];
			}
		}
		if (instance.classAttribute().isNumeric() == true) {
			if (numPreds == 0) {
				sums[0] = Utils.missingValue();
			} else {
				sums[0] /= numPreds;
			}
			return sums;
		} else if (Utils.eq(Utils.sum(sums), 0)) {
			return sums;
		} else {
			Utils.normalize(sums);
			return sums;
		}
	}

	private int getMinorityClass(Instances data) {
		int minIndex = 0;
		int min = Integer.MAX_VALUE;
		// find minority class
		int[] classCounts = data.attributeStats(data.classIndex()).nominalCounts;
		for (int i = 0; i < classCounts.length; i++) {
			if (classCounts[i] != 0 && classCounts[i] < min) {
				min = classCounts[i];
				minIndex = i;
			}
		}
		return minIndex;
	}

	/**
	 * Returns a training set for a particular partition.
	 * 
	 * @param partition
	 *            the number of the partition for the requested training set.
	 * @return the training set for the supplied iteration number
	 * @throws Exception
	 *             if something goes wrong when generating a training set.
	 */

	@Override
	protected synchronized Instances getTrainingSet(int partition) throws Exception {
		Instances trainingSet = m_majorityData.testCV(getNumIterations(), partition);
		trainingSet.addAll(m_minorityData);
		return trainingSet;
	}

}
