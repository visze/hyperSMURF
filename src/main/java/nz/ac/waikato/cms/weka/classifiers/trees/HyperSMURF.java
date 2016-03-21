package nz.ac.waikato.cms.weka.classifiers.trees;

import java.util.Random;

import nz.ac.waikato.cms.weka.classifiers.meta.EasyEnsemble;
import weka.classifiers.Classifier;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.RandomizableFilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.Randomizable;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.supervised.instance.SpreadSubsample;
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
public class HyperSMURF extends EasyEnsemble {

	/** for serialization */
	private static final long serialVersionUID = -4869310424420765879L;

	protected Classifier m_Classifier = new RandomizableFilteredClassifier();

	protected Classifier m_BaseClassifier = getRandomForest();

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
	 * Set the base learner. Replaces only the base-learner (Random Forest) not the SMOTE and subsampling.
	 *
	 * @param newClassifier
	 *            the classifier to use.
	 */
	@Override
	public void setClassifier(Classifier newClassifier) {

		m_BaseClassifier = newClassifier;

	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation result = new TechnicalInformation(Type.UNPUBLISHED);

		result.setValue(Field.AUTHOR, "Schubach M, Robinson PN, and Valentini G");

		return result;
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
		
		m_Classifier = getFilteredClassifier();
		
		this.buildEasyEnsembleClassifier();

	}

	private Classifier getFilteredClassifier() throws Exception {
		
		MultiFilter mfilter = new MultiFilter();
		mfilter.setDebug(m_Debug);
		mfilter.setDoNotCheckCapabilities(m_DoNotCheckCapabilities);
		mfilter.setFilters(new Filter[]{getSMOTE(),getSpreadSubsample()});
		mfilter.setInputFormat(m_data);
		
		RandomizableFilteredClassifier classifier = new RandomizableFilteredClassifier();
		classifier.setClassifier(getRandomForest());
		classifier.setNumDecimalPlaces(m_numDecimalPlaces);
		classifier.setDebug(m_Debug);
		classifier.setBatchSize(m_BatchSize);
		classifier.setDoNotCheckCapabilities(m_DoNotCheckCapabilities);
		classifier.setSeed(m_Seed);
		classifier.setFilter(mfilter);
		classifier.setClassifier(getRandomForest());
		
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
		// TODO SMOTE MUST ALWAYS HAVE ANOTHER SEED!
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
