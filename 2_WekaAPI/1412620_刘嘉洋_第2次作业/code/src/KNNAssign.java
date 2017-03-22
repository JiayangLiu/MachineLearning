import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;

/**
 * MachineLearning Lab2, Assignment 4: KNN comparison
 * 
 * @author LIU Jiayang. 1412620. NKCS
 * @date Tue 7 Mar 2017
 */
public class KNNAssign {
	public static Instances getFileInstances(String fileName) throws Exception {
		FileReader frData = new FileReader(fileName);
		Instances data = new Instances(frData);
		return data;
	}

	/**
	 * 一致评估方法。评价当前数据集及分类器情况下在十折交叉认证中的正确率。
	 * @param instances 数据实例
	 * @param classifier 采用分类器
	 * @throws Exception
	 */
	private static void unifiedEvaluation(Instances instances, Classifier classifier) throws Exception {
		Evaluation eval = new Evaluation(instances);
		eval.crossValidateModel(classifier, instances, 10, new Random(1)); // 使用10折交叉认证
		System.out.println(1 - eval.errorRate()); // 正确率 = 1 - 错误率
	}

	public static void main(String[] args) throws Exception {
		Instances instances = getFileInstances(
				"/Users/macdowell/Desktop/Professional Materials/3_Junior_S2/2_MachineLearning/Lab/2_WekaAPI/实验二 数据集/bank.arff");
		instances.setClassIndex(instances.numAttributes() - 1); // 设置最后一列为类别

		System.out.println("------------------K = 1------------------");
		Classifier classifier1 = new IBk();
		((IBk) classifier1).setKNN(1);
		unifiedEvaluation(instances, classifier1);
		System.out.println("\n------------------K = 5------------------");
		Classifier classifier2 = new IBk();
		((IBk) classifier2).setKNN(5);
		unifiedEvaluation(instances, classifier2);
		System.out.println("\n------------------K = 10------------------");
		Classifier classifier3 = new IBk();
		((IBk) classifier3).setKNN(10);
		unifiedEvaluation(instances, classifier3);
		System.out.println("\n------------------K = 30------------------");
		Classifier classifier4 = new IBk();
		((IBk) classifier4).setKNN(30);
		unifiedEvaluation(instances, classifier4);
		
		System.out.println("------------------K连续取值------------------");
		for (int i=1; i<=230; i++) {
			Classifier classifier5 = new IBk();
			((IBk) classifier5).setKNN(i);
			System.out.print("K = " + i + ", accuracy = ");
			unifiedEvaluation(instances, classifier5);
		}
	}
}
