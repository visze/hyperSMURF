package weka.classifiers.meta;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import weka.classifiers.RandomizableParallelIteratedSingleClassifierEnhancer;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
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
 *    Journal = {IEEE Transactions on Systems, Man, and Cybernetics, Part B: Cybernetics},
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
 *  -I &lt;num&gt;
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
public class EasyEnsemble extends RandomizableParallelIteratedSingleClassifierEnhancer
		implements TechnicalInformationHandler {

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
	 * Returns a string describing classifier
	 * 
	 * @return a description suitable for displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return "Class for EasyEnsemble, a classifier for imbalanced datasets. Can do classification "
				+ "and regression depending on the base learner. \n\n" + "For more information, see\n\n"
				+ getTechnicalInformation().toString();
	}

	/**
	 * Returns an instance of a TechnicalInformation object, containing detailed information about the technical
	 * background of this class, e.g., paper reference or book this class is based on.
	 * 
	 * @return the technical information about this class
	 */
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result;

		result = new TechnicalInformation(Type.ARTICLE);
		result.setValue(Field.AUTHOR, "Xu-Ying Liu");
		result.setValue(Field.YEAR, "2009");
		result.setValue(Field.TITLE, "Exploratory Undersampling for Class-Imbalance Learning");
		result.setValue(Field.JOURNAL, "IEEE Transactions on Systems, Man, and Cybernetics, Part B: Cybernetics");
		result.setValue(Field.VOLUME, "39");
		result.setValue(Field.NUMBER, "2");
		result.setValue(Field.PAGES, "539-550");

		return result;
	}

	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>(0);

		newVector.addAll(Collections.list(super.listOptions()));

		return newVector.elements();
	}

	/**
	 * Parses a given list of options.
	 * <p/>
	 *
	 * <!-- options-start --> Valid options are:
	 * <p/>
	 * 
	 * <pre>
	 *  -I &lt;num&gt;
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
	 * @param options
	 *            the list of options as an array of strings
	 * @throws Exception
	 *             if an option is not supported
	 */
	@Override
	public void setOptions(String[] options) throws Exception {

		super.setOptions(options);

		Utils.checkForRemainingOptions(options);
	}

	/**
	 * Gets the current settings of the Classifier.
	 *
	 * @return an array of strings suitable for passing to setOptions
	 */
	@Override
	public String[] getOptions() {

		Vector<String> options = new Vector<String>();

		options.add("-P");
		options.add("" + getNumIterations());

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[0]);
	}

	@Override
	public String numIterationsTipText() {
		return "The number of partitions to be used.";
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
		
		m_random = new Random(m_Seed);
		
		super.buildClassifier(m_data);
		
		this.buildEasyEnsembleClassifier();
		
		

	}
	
	/**
	 * outsource easy-ensemble specific buildClassifier methods for better extension of this method
	 * 
	 * @throws Exception
	 */
	protected void  buildEasyEnsembleClassifier() throws Exception {
		

		RemoveWithValues classValueFilter = new RemoveWithValues();
		classValueFilter.setAttributeIndex(Integer.toString(m_data.classIndex() + 1));
		classValueFilter.setNominalIndicesArr(new int[] { getMinorityClass(m_data) + 1 });
		classValueFilter.setInputFormat(m_data);

		m_minorityData = Filter.useFilter(m_data, classValueFilter);

		classValueFilter.setInvertSelection(true);
		classValueFilter.setInputFormat(m_data);
		m_majorityData = Filter.useFilter(m_data, classValueFilter);

		// save memory
		m_data = null;

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

	@Override
	public String toString() {

		if (m_Classifiers == null)
			return "EasyEnsemble: No model built yet.";

		StringBuffer text = new StringBuffer();
		text.append("All the base classifiers: \n\n");
		for (int i = 0; i < m_Classifiers.length; i++)
			text.append(m_Classifiers[i].toString() + "\n\n");

		return text.toString();
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
