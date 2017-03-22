import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * MachineLearning Lab2, Assignment 3_2: Classifier with trainingSet & testingSet
 * 
 * @author LIU Jiayang. 1412620. NKCS
 * @date Tue 7 Mar 2017
 */
public class ClassifierAssign2 {
	public static Instances getFileInstances(String fileName) throws Exception {
		FileReader frData = new FileReader(fileName);
		Instances data = new Instances(frData);
		return data;
	}

	/**
	 * 一致评估方法。评价当前数据训练集及分类器情况下在测试集中的正确率。
	 * @param instancesTrain 数据训练实例
	 * @param instancesTrain 数据测试实例
	 * @param classifier 采用分类器
	 * @throws Exception
	 */
	private static void unifiedEvaluation(Instances instancesTrain, Instances instancesTest, Classifier classifier) throws Exception {
		classifier.buildClassifier(instancesTrain); // 通过训练集训练分类模型
		Evaluation eval = new Evaluation(instancesTrain);
		eval.evaluateModel(classifier, instancesTest);	// 通过测试集测试分类模型
		System.out.println(1 - eval.errorRate()); // 正确率 = 1 - 错误率
	}

	public static void main(String[] args) throws Exception {
		String basedPath = "/Users/macdowell/Desktop/Professional Materials/3_Junior_S2/2_MachineLearning/Lab/2_WekaAPI/实验二 数据集/";
		DataSource trainingSet=new DataSource( basedPath + "U_segmentation_train.arff");
		Instances instancesTrain = trainingSet.getDataSet();
		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1); // 设置最后一列为类别
		
		DataSource testingSet = new DataSource( basedPath + "U_segmentation_test.arff");
		Instances instanceTest = testingSet.getDataSet();
		instanceTest.setClassIndex(instanceTest.numAttributes() - 1); // 设置最后一列为类别
		
		System.out.println("------------------使用朴素贝叶斯分类器------------------");
		Classifier classifier1 = new NaiveBayes();
		unifiedEvaluation(instancesTrain, instanceTest, classifier1);
		System.out.println("\n------------------使用SMO分类器------------------");
		Classifier classifier2 = new SMO();
		unifiedEvaluation(instancesTrain, instanceTest, classifier2);
		System.out.println("\n------------------使用J48分类器------------------");
		Classifier classifier3 = new J48();
		unifiedEvaluation(instancesTrain, instanceTest, classifier3);
		System.out.println("\n------------------使用1NN分类器------------------");
		Classifier classifier4 = new IBk();
		((IBk) classifier4).setKNN(1);
		unifiedEvaluation(instancesTrain, instanceTest, classifier4);
	}
}
