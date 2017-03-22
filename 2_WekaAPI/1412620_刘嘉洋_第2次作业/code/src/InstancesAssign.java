import java.io.FileReader;
import weka.core.Instances;

/**
 * MachineLearning Lab2, Assignment 1: Instances
 * 
 * @author LIU Jiayang. 1412620. NKCS
 * @date Tue 7 Mar 2017
 */
public class InstancesAssign {
	public static Instances getFileInstances(String fileName) throws Exception {
		FileReader frData = new FileReader(fileName);
		Instances data = new Instances(frData);
		return data;
	}

	public static void main(String[] args) throws Exception {
		Instances instances = getFileInstances(
				"/Users/macdowell/Desktop/Professional Materials/3_Junior_S2/2_MachineLearning/Lab/2_WekaAPI/实验二 数据集/bank.arff");
		// 注意：要首先设置类属性标签，不然会在调用numClasses()时报"UnassignedClassException"
		instances.setClassIndex(instances.numAttributes() - 1); // 设置最后一列为类别

		System.out.println("样例数: " + instances.numInstances());
		System.out.println("属性数: " + instances.numAttributes());
		System.out.println("类别数: " + instances.numClasses());
	}
}
