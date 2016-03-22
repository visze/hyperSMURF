package nz.ac.waikato.cms.weka.classifiers.trees;

import java.util.Collections;
import java.util.Enumeration;
import java.util.Random;
import java.util.Vector;

import nz.ac.waikato.cms.weka.classifiers.meta.EasyEnsemble;
import weka.classifiers.Classifier;
import weka.classifiers.RandomizableClassifier;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.RandomizableFilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.SpreadSubsample;

/**
 * <!-- globalinfo-start -->
 * 
 * Class for Hyper SMOTE Undersampling with Random Forests (HyperSMURF) classifier. EasyEnsemble splits up the majority
 * class into n partitions and trains n RandomForest using (a downsampled) majority partition and all upsampled (SMOTE)
 * minority classes.<br/>
 * <br/>
 * For more information, see<br/>
 * <br/>
 * Schubach M, Robinson PN, Valentini G. Unpublished.
 * <p/>
 * <!-- globalinfo-end -->
 *
 * <!-- technical-bibtex-start --> BibTeX:
 * 
 * <pre>
 * &#64;article{Schubach,
 *    Author = {Schubach, M. and Robinson, PN. and Valentini, G.},
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 *
 * <!-- options-start -->
 * 
 * Valid options are:
 * <p/>
 * 
 * <pre>
 *  -S &lt;num&gt;
 *  Random number seed.
 *  (default 1)
 * </pre>
 * 
 * <pre>
 *  -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console
 * </pre>
 * 
 * <pre>
 *  -B
 *  Full name of base classifier.
 *  (default: weka.classifiers.trees.RandomForest)
 * </pre>
 * 
 * <p/>
 * EasyEnsemble:
 * 
 * <pre>
 *  -I &lt;num&gt;
 *  Number of partitions of the majority class. 
 *  (default 10)
 * </pre>
 * 
 * 
 * <pre>
 *  -num-slots &lt;num&gt;
 *  Number of execution slots.
 *  (default 1 - i.e. no parallelism)
 * </pre>
 * 
 * <p/>
 * SMOTE:
 * 
 * <pre>
 *  -P &lt;percentage&gt;
 *  Specifies percentage of SMOTE instances to create.
 *  (default 100.0)
 * </pre>
 * 
 * <pre>
 *  -K &lt;nearest-neighbors&gt;
 *  Specifies the number of nearest neighbors to use.
 *  (default 5)
 * </pre>
 * 
 * <pre>
 *  -C &lt;value-index&gt;
 *  Specifies the index of the nominal class value to SMOTE
 *  (default 0: auto-detect non-empty minority class))
 * </pre>
 * 
 * <p/>
 * SpreadSubsample
 * 
 * <pre>
 * -M &lt;num&gt;
 *  The maximum class distribution spread.
 *  0 = no maximum spread, 1 = uniform distribution, 10 = allow at most
 *  a 10:1 ratio between the classes (default 0)
 * </pre>
 * 
 * <pre>
 * -A
 *  Adjust weights so that total weight per class is maintained.
 *  Individual instance weighting is not preserved. (default no
 *  weights adjustment)
 * </pre>
 * 
 * <pre>
 * -X &lt;num&gt;
 *  The maximum count for any class value (default 0 = unlimited).
 * </pre>
 * 
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
public class HyperSMURF extends EasyEnsemble {

	/** for serialization */
	private static final long serialVersionUID = -4869310424420765879L;

	protected Classifier m_Classifier = null;

	// RandomForest

	/** Number of trees in forest. */
	protected int m_numTrees = 10;

	/**
	 * Number of features to consider in random feature selection. If less than 1 will use int(log_2(M)+1) )
	 */
	protected int m_numFeatures = 0;

	/** Final number of features that were considered in last build. */
	protected int m_KValue = 0;

	/** The bagger. */
	protected Bagging m_bagger = null;

	/** The maximum depth of the trees (0 = unlimited) */
	protected int m_MaxDepth = 0;

	/** The number of threads to have executing at any one time */
	protected int m_numRFExecutionSlots = 1;

	/** Print the individual trees in the output */
	protected boolean m_printTrees = false;

	/** Don't calculate the out of bag error */
	protected boolean m_dontCalculateOutOfBagError;

	/** Whether to break ties randomly. */
	protected boolean m_BreakTiesRandomly = false;

	// SpreadSubsample

	/** The maximum count of any class */
	protected int m_MaxCount = 0;

	/** True if the first batch has been done */
	protected double m_DistributionSpread = 0;

	/**
	 * True if instance weights will be adjusted to maintain total weight per class.
	 */
	protected boolean m_AdjustWeights = false;

	// SMOTE

	/** the number of neighbors to use. */
	protected int m_NearestNeighbors = 5;

	/** the percentage of SMOTE instances to create. */
	protected double m_Percentage = 100.0;

	/** the index of the class value. */
	protected String m_ClassValueIndex = "0";

	/** whether to detect the minority class automatically. */
	protected boolean m_DetectMinorityClass = true;

	/**
	 * Returns a string describing classifier
	 * 
	 * @return a description suitable for displaying in the explorer/experimenter gui
	 */
	public String globalInfo() {

		return "Class for constructing a hyperSMURF.\n\n" + "For more information see: \n\n"
				+ getTechnicalInformation().toString();
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result = new TechnicalInformation(Type.UNPUBLISHED);

		result.setValue(Field.AUTHOR, "Schubach M, Robinson PN, and Valentini G");

		return result;
	}

	@Override
	public Enumeration<Option> listOptions() {

		Vector<Option> newVector = new Vector<Option>(6);
		// SMOTE
		newVector
				.addElement(new Option("\tSpecifies percentage of SMOTE instances to create.\n" + "\t(default 100.0)\n",
						"P", 1, "-P <percentage>"));
		newVector.addElement(new Option("\tSpecifies the number of nearest neighbors to use.\n" + "\t(default 5)\n",
				"K", 1, "-K <nearest-neighbors>"));
		newVector
				.addElement(new Option(
						"\tSpecifies the index of the nominal class value to SMOTE\n"
								+ "\t(default 0: auto-detect non-empty minority class))\n",
						"C", 1, "-C <value-index>"));
		// SpreadSubsample
		newVector.addElement(new Option("\tThe maximum class distribution spread.\n"
				+ "\t0 = no maximum spread, 1 = uniform distribution, 10 = allow at most\n"
				+ "\ta 10:1 ratio between the classes (default 0)", "M", 1, "-M <num>"));
		newVector.addElement(new Option(
				"\tAdjust weights so that total weight per class is maintained.\n"
						+ "\tIndividual instance weighting is not preserved. (default no\n" + "\tweights adjustment",
				"A", 0, "-A"));
		newVector.addElement(
				new Option("\tThe maximum count for any class value (default 0 = unlimited).\n", "X", 0, "-X <num>"));

		// others
		newVector.addAll(Collections.list(super.listOptions()));
		return newVector.elements();
	}

	@Override
	public void setOptions(String[] options) throws Exception {

		String percentageStr = Utils.getOption('P', options);
		if (percentageStr.length() != 0) {
			setPercentage(new Double(percentageStr).doubleValue());
		} else {
			setPercentage(100.0);
		}

		String nnStr = Utils.getOption('K', options);
		if (nnStr.length() != 0) {
			setNearestNeighbors(Integer.parseInt(nnStr));
		} else {
			setNearestNeighbors(5);
		}

		String classValueIndexStr = Utils.getOption('C', options);
		if (classValueIndexStr.length() != 0) {
			setClassValue(classValueIndexStr);
		} else {
			m_DetectMinorityClass = true;
		}

		String maxString = Utils.getOption('M', options);
		if (maxString.length() != 0) {
			setDistributionSpread(Double.valueOf(maxString).doubleValue());
		} else {
			setDistributionSpread(0);
		}

		String maxCount = Utils.getOption('X', options);
		if (maxCount.length() != 0) {
			setMaxCount(Double.valueOf(maxCount).doubleValue());
		} else {
			setMaxCount(0);
		}

		m_AdjustWeights = Utils.getFlag('A', options);

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

		options.add("-C");
		options.add(getClassValue());

		options.add("-K");
		options.add("" + getNearestNeighbors());

		options.add("-P");
		options.add("" + getPercentage());

		options.add("-M");
		options.add("" + getDistributionSpread());

		options.add("-X");
		options.add("" + getMaxCount());

		if (getAdjustWeights()) {
			options.add("-W");
		}

		Collections.addAll(options, super.getOptions());

		return options.toArray(new String[options.size()]);
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String numTreesTipText() {
		return "The number of trees to be generated.";
	}

	/**
	 * Get the value of numTrees.
	 * 
	 * @return Value of numTrees.
	 */
	public int getNumTrees() {

		return m_numTrees;
	}

	/**
	 * Set the value of numTrees.
	 * 
	 * @param newNumTrees
	 *            Value to assign to numTrees.
	 */
	public void setNumTrees(int newNumTrees) {

		m_numTrees = newNumTrees;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String numFeaturesTipText() {
		return "The number of attributes to be used in random selection (see RandomTree).";
	}

	/**
	 * Get the number of features used in random selection.
	 * 
	 * @return Value of numFeatures.
	 */
	public int getNumFeatures() {

		return m_numFeatures;
	}

	/**
	 * Set the number of features to use in random selection.
	 * 
	 * @param newNumFeatures
	 *            Value to assign to numFeatures.
	 */
	public void setNumFeatures(int newNumFeatures) {

		m_numFeatures = newNumFeatures;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String maxDepthTipText() {
		return "The maximum depth of the trees, 0 for unlimited.";
	}

	/**
	 * Get the maximum depth of trh tree, 0 for unlimited.
	 * 
	 * @return the maximum depth.
	 */
	public int getMaxDepth() {
		return m_MaxDepth;
	}

	/**
	 * Set the maximum depth of the tree, 0 for unlimited.
	 * 
	 * @param value
	 *            the maximum depth.
	 */
	public void setMaxDepth(int value) {
		m_MaxDepth = value;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String printTreesTipText() {
		return "Print the individual trees in the output";
	}

	/**
	 * Set whether to print the individual ensemble trees in the output
	 * 
	 * @param print
	 *            true if the individual trees are to be printed
	 */
	public void setPrintTrees(boolean print) {
		m_printTrees = print;
	}

	/**
	 * Get whether to print the individual ensemble trees in the output
	 * 
	 * @return true if the individual trees are to be printed
	 */
	public boolean getPrintTrees() {
		return m_printTrees;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String dontCalculateOutOfBagErrorTipText() {
		return "If true, then the out of bag error is not computed";
	}

	/**
	 * Set whether to turn off the calculation of out of bag error
	 * 
	 * @param b
	 *            true to turn off the calculation of out of bag error
	 */
	public void setDontCalculateOutOfBagError(boolean b) {
		m_dontCalculateOutOfBagError = b;
	}

	/**
	 * Get whether to turn off the calculation of out of bag error
	 * 
	 * @return true to turn off the calculation of out of bag error
	 */
	public boolean getDontCalculateOutOfBagError() {
		return m_dontCalculateOutOfBagError;
	}

	/**
	 * Gets the out of bag error that was calculated as the classifier was built.
	 * 
	 * @return the out of bag error
	 */
	public double measureOutOfBagError() {

		if (m_bagger != null && !m_dontCalculateOutOfBagError) {
			return m_bagger.measureOutOfBagError();
		} else {
			return Double.NaN;
		}
	}

	/**
	 * Set the number of execution slots (threads) to use for building the members of the ensemble.
	 * 
	 * @param numSlots
	 *            the number of slots to use.
	 */
	public void setNumRFExecutionSlots(int numSlots) {
		m_numRFExecutionSlots = numSlots;
	}

	/**
	 * Get the number of execution slots (threads) to use for building the members of the ensemble.
	 * 
	 * @return the number of slots to use
	 */
	public int getNumRFExecutionSlots() {
		return m_numRFExecutionSlots;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String numRFExecutionSlotsTipText() {
		return "The number of execution slots (threads) to use for " + "constructing the ensemble.";
	}

	/**
	 * Returns the tip text for this property
	 *
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String breakTiesRandomlyTipText() {
		return "Break ties randomly when several attributes look equally good.";
	}

	/**
	 * Get whether to break ties randomly.
	 *
	 * @return true if ties are to be broken randomly.
	 */
	public boolean getBreakTiesRandomly() {

		return m_BreakTiesRandomly;
	}

	/**
	 * Set whether to break ties randomly.
	 *
	 * @param newBreakTiesRandomly
	 *            true if ties are to be broken randomly
	 */
	public void setBreakTiesRandomly(boolean newBreakTiesRandomly) {

		m_BreakTiesRandomly = newBreakTiesRandomly;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String percentageTipText() {
		return "The percentage of SMOTE instances to create.";
	}

	/**
	 * Sets the percentage of SMOTE instances to create.
	 * 
	 * @param value
	 *            the percentage to use
	 */
	public void setPercentage(double value) {
		if (value >= 0)
			m_Percentage = value;
		else
			System.err.println("Percentage must be >= 0!");
	}

	/**
	 * Gets the percentage of SMOTE instances to create.
	 * 
	 * @return the percentage of SMOTE instances to create
	 */
	public double getPercentage() {
		return m_Percentage;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String nearestNeighborsTipText() {
		return "The number of nearest neighbors to use.";
	}

	/**
	 * Sets the number of nearest neighbors to use.
	 * 
	 * @param value
	 *            the number of nearest neighbors to use
	 */
	public void setNearestNeighbors(int value) {
		if (value >= 1)
			m_NearestNeighbors = value;
		else
			System.err.println("At least 1 neighbor necessary!");
	}

	/**
	 * Gets the number of nearest neighbors to use.
	 * 
	 * @return the number of nearest neighbors to use
	 */
	public int getNearestNeighbors() {
		return m_NearestNeighbors;
	}

	/**
	 * Returns the tip text for this property.
	 * 
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String classValueTipText() {
		return "The index of the class value to which SMOTE should be applied. "
				+ "Use a value of 0 to auto-detect the non-empty minority class.";
	}

	/**
	 * Sets the index of the class value to which SMOTE should be applied.
	 * 
	 * @param value
	 *            the class value index
	 */
	public void setClassValue(String value) {
		m_ClassValueIndex = value;
		if (m_ClassValueIndex.equals("0")) {
			m_DetectMinorityClass = true;
		} else {
			m_DetectMinorityClass = false;
		}
	}

	/**
	 * Gets the index of the class value to which SMOTE should be applied.
	 * 
	 * @return the index of the clas value to which SMOTE should be applied
	 */
	public String getClassValue() {
		return m_ClassValueIndex;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String adjustWeightsTipText() {
		return "Wether instance weights will be adjusted to maintain total weight per " + "class.";
	}

	/**
	 * Returns true if instance weights will be adjusted to maintain total weight per class.
	 * 
	 * @return true if instance weights will be adjusted to maintain total weight per class.
	 */
	public boolean getAdjustWeights() {

		return m_AdjustWeights;
	}

	/**
	 * Sets whether the instance weights will be adjusted to maintain total weight per class.
	 * 
	 * @param newAdjustWeights
	 *            whether to adjust weights
	 */
	public void setAdjustWeights(boolean newAdjustWeights) {

		m_AdjustWeights = newAdjustWeights;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String distributionSpreadTipText() {
		return "The maximum class distribution spread. "
				+ "(0 = no maximum spread, 1 = uniform distribution, 10 = allow at most a "
				+ "10:1 ratio between the classes).";
	}

	/**
	 * Sets the value for the distribution spread
	 * 
	 * @param spread
	 *            the new distribution spread
	 */
	public void setDistributionSpread(double spread) {

		m_DistributionSpread = spread;
	}

	/**
	 * Gets the value for the distribution spread
	 * 
	 * @return the distribution spread
	 */
	public double getDistributionSpread() {

		return m_DistributionSpread;
	}

	/**
	 * Returns the tip text for this property
	 * 
	 * @return tip text for this property suitable for displaying in the explorer/experimenter gui
	 */
	public String maxCountTipText() {
		return "The maximum count for any class value (0 = unlimited).";
	}

	/**
	 * Sets the value for the max count
	 * 
	 * @param maxcount
	 *            the new max count
	 */
	public void setMaxCount(double maxcount) {

		m_MaxCount = (int) maxcount;
	}

	/**
	 * Gets the value for the max count
	 * 
	 * @return the max count
	 */
	public double getMaxCount() {

		return m_MaxCount;
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

		m_Classifiers = new Classifier[m_NumIterations];
		for (int i = 0; i < m_Classifiers.length; i++) {
			m_Classifiers[i] = getFilteredClassifier();
		}

		if (m_numExecutionSlots < 0) {
			throw new Exception("Number of execution slots needs to be >= 0!");
		}

		this.buildEasyEnsembleClassifier();

	}

	private Classifier getFilteredClassifier() throws Exception {

		MultiFilter mfilter = new MultiFilter();
		mfilter.setDebug(m_Debug);
		mfilter.setDoNotCheckCapabilities(m_DoNotCheckCapabilities);
		mfilter.setFilters(new Filter[] { getSMOTE(), getSpreadSubsample() });
		mfilter.setInputFormat(m_data);

		FilteredClassifier classifier;

		// Set the random forest as base learner if no other method is set.
		if (m_Classifier == null || m_Classifier instanceof RandomizableClassifier) {
			classifier = new RandomizableFilteredClassifier();
			((RandomizableFilteredClassifier) classifier).setSeed(m_Seed);

		} else
			classifier = new FilteredClassifier();

		classifier.setNumDecimalPlaces(m_numDecimalPlaces);
		classifier.setDebug(m_Debug);
		classifier.setBatchSize(m_BatchSize);
		classifier.setDoNotCheckCapabilities(m_DoNotCheckCapabilities);
		classifier.setFilter(mfilter);
		if (m_Classifier == null)
			classifier.setClassifier(getRandomForest());
		else
			classifier.setClassifier(m_Classifier);

		return classifier;
	}

	private Filter getSpreadSubsample() throws Exception {
		SpreadSubsample subsample = new SpreadSubsample();
		subsample.setDistributionSpread(m_DistributionSpread);
		subsample.setAdjustWeights(m_AdjustWeights);
		subsample.setMaxCount(m_MaxCount);
		subsample.setRandomSeed(m_random.nextInt());
		subsample.setDebug(m_Debug);
		subsample.setDoNotCheckCapabilities(m_DoNotCheckCapabilities);
		subsample.setInputFormat(m_data);
		return subsample;
	}

	private Filter getSMOTE() throws Exception {
		SMOTE smote = new SMOTE();
		smote.setPercentage(m_Percentage);
		smote.setNearestNeighbors(m_NearestNeighbors);
		smote.setRandomSeed(m_random.nextInt());
		smote.setClassValue(m_ClassValueIndex);
		smote.setDoNotCheckCapabilities(m_DoNotCheckCapabilities);
		smote.setDebug(m_Debug);
		smote.setInputFormat(m_data);
		return smote;
	}

	private RandomForest getRandomForest() {
		RandomForest randomForest = new RandomForest();
		randomForest.setBatchSize(m_BatchSize);
		randomForest.setBreakTiesRandomly(m_BreakTiesRandomly);
		randomForest.setDontCalculateOutOfBagError(m_dontCalculateOutOfBagError);
		randomForest.setMaxDepth(m_MaxDepth);
		randomForest.setNumDecimalPlaces(m_numDecimalPlaces);
		randomForest.setNumExecutionSlots(m_numRFExecutionSlots);
		randomForest.setNumFeatures(m_numFeatures);
		randomForest.setNumTrees(m_numTrees);
		randomForest.setPrintTrees(m_printTrees);
		randomForest.setSeed(m_random.nextInt());
		randomForest.setDoNotCheckCapabilities(m_DoNotCheckCapabilities);
		randomForest.setDebug(m_Debug);
		return randomForest;
	}

}
