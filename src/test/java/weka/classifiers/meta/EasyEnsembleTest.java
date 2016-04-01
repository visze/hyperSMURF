package weka.classifiers.meta;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Random;

import org.junit.BeforeClass;
import org.junit.Test;

import com.google.common.io.Resources;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;

public class EasyEnsembleTest {

	private static Instances data;
	private static Instances randData;
	private static String diabetesFile = "diabetes.arff";
	private static int seed = 42;
	private int folds = 10;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		File file = new File(Resources.getResource(diabetesFile).toString());
		BufferedReader reader = new BufferedReader(new FileReader(file));
		data = new Instances(reader);
		reader.close();
		// setting class attribute
		data.setClassIndex(data.numAttributes() - 1);
		Random rand = new Random(seed); // create seeded number generator
		randData = new Instances(data); // create copy of original data
		randData.randomize(rand);
	}

	@Test
	public void classifyJ48Test() throws Exception {

		EasyEnsemble easyEnsemble = new EasyEnsemble();
		easyEnsemble.setClassifier(new J48());

			
			Evaluation eval = new Evaluation(randData);
			eval.crossValidateModel(easyEnsemble, randData, folds, new Random(seed));


	}

}
