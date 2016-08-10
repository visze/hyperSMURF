package weka.classifiers.trees;

import static org.junit.Assert.assertThat;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.Random;
import java.util.zip.GZIPInputStream;

import org.hamcrest.Matchers;
import org.junit.Before;
import org.junit.Test;

import com.google.common.io.Resources;

import weka.classifiers.Evaluation;
import weka.core.Instances;

public class HyperSMURFTest {

	private Instances randDiabetesData;
	private Instances randGeneratedImbalancedBinData;
	private Instances randGeneratedImbalancedData;
	private String diabetesFile = "diabetes.arff.gz";
	private String generatedImbalancedBinFile = "randomImbalancedBinDataset.arff.gz";
	private String generatedImbalancedFile = "randomImbalancedDataset.arff.gz";
	private int seed = 42;
	private int folds = 10;

	@Before
	public void setUp() throws Exception {
		File file = new File(Resources.getResource(diabetesFile).getPath());
		GZIPInputStream in = new GZIPInputStream(new FileInputStream(file));
		BufferedReader reader = new BufferedReader(new InputStreamReader(in));
		randDiabetesData = new Instances(reader);
		in.close();
		reader.close();
		// setting class attribute
		randDiabetesData.setClassIndex(randDiabetesData.numAttributes() - 1);
		Random rand = new Random(seed); // create seeded number generator
		randDiabetesData.randomize(rand);

		file = new File(Resources.getResource(generatedImbalancedBinFile).getPath());
		in = new GZIPInputStream(new FileInputStream(file));
		reader = new BufferedReader(new InputStreamReader(in));
		randGeneratedImbalancedBinData = new Instances(reader);
		in.close();
		reader.close();
		// setting class attribute
		randGeneratedImbalancedBinData.setClassIndex(randGeneratedImbalancedBinData.numAttributes() - 1);
		randGeneratedImbalancedBinData.randomize(rand);

		file = new File(Resources.getResource(generatedImbalancedFile).getPath());
		in = new GZIPInputStream(new FileInputStream(file));
		reader = new BufferedReader(new InputStreamReader(in));
		randGeneratedImbalancedData = new Instances(reader);
		in.close();
		reader.close();
		// setting class attribute
		randGeneratedImbalancedData.setClassIndex(randGeneratedImbalancedData.numAttributes() - 1);
		randGeneratedImbalancedData.randomize(rand);
	}

	@Test
	public void classifyJ48Test() throws Exception {

		HyperSMURF hyperSMURF = new HyperSMURF();
		hyperSMURF.setNumIterations(2);
		hyperSMURF.setClassifier(new J48());

		Evaluation eval = new Evaluation(randDiabetesData);
		eval.crossValidateModel(hyperSMURF, randDiabetesData, folds, new Random(seed));

		double prcHyperSMURF = eval.areaUnderPRC(1);
		double rocHyperSMURF = eval.areaUnderROC(1);

		eval = new Evaluation(randDiabetesData);
		eval.crossValidateModel(new J48(), randDiabetesData, folds, new Random(seed));
		double prcJ48 = eval.areaUnderPRC(1);
		double rocJ48 = eval.areaUnderROC(1);

		assertThat(prcHyperSMURF, Matchers.greaterThan(prcJ48));
		assertThat(rocHyperSMURF, Matchers.greaterThan(rocJ48));
	}

	@Test
	public void classifyRFRandomBinDataTest() throws Exception {

		HyperSMURF hyperSMURF = new HyperSMURF();
		hyperSMURF.setNumIterations(10);
		hyperSMURF.setNumExecutionSlots(10);
		hyperSMURF.setClassifier(new J48());

		Evaluation eval = new Evaluation(randGeneratedImbalancedBinData);
		eval.crossValidateModel(hyperSMURF, randGeneratedImbalancedBinData, folds, new Random(seed));

		double prcHyperSMURF = eval.areaUnderPRC(0);
		double rocHyperSMURF = eval.areaUnderROC(0);
		eval = new Evaluation(randGeneratedImbalancedBinData);
		eval.crossValidateModel(new J48(), randGeneratedImbalancedBinData, folds, new Random(seed));
		double prcJ48 = eval.areaUnderPRC(0);
		double rocJ48 = eval.areaUnderROC(0);

		assertThat(prcHyperSMURF, Matchers.greaterThan(prcJ48));
		assertThat(rocHyperSMURF, Matchers.greaterThan(rocJ48));
	}

	@Test
	public void classifyRFRandomDataTest() throws Exception {

		HyperSMURF hyperSMURF = new HyperSMURF();
		hyperSMURF.setNumIterations(50);
		hyperSMURF.setPercentage(200.0);
		hyperSMURF.setNumExecutionSlots(10);

		Evaluation eval = new Evaluation(randGeneratedImbalancedData);
		eval.crossValidateModel(hyperSMURF, randGeneratedImbalancedData, folds, new Random(seed));

		double prcHyperSMURF = eval.areaUnderPRC(1);
		double rocHyperSMURF = eval.areaUnderROC(1);

		eval = new Evaluation(randGeneratedImbalancedData);
		RandomForest rf = new RandomForest();
		rf.setNumExecutionSlots(10);
		rf.setNumIterations(10);
		eval.crossValidateModel(rf, randGeneratedImbalancedData, folds, new Random(seed));
		double prcRF = eval.areaUnderPRC(1);
		double rocRF = eval.areaUnderROC(1);

		assertThat(prcHyperSMURF, Matchers.greaterThan(prcRF));
		assertThat(rocHyperSMURF, Matchers.greaterThan(rocRF));
	}

}
