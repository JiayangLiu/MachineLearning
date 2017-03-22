import java.io.FileReader;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Liu_Test {
	public static Instances getFileInstances(String fileName) throws Exception {
		FileReader frData = new FileReader(fileName);
		Instances data = new Instances(frData);
		return data;
	}
	
	//训练集与测试集评估
	private static void evaluateNaiveBayes(Instances dataTrain,Instances dataTest)throws Exception{
		Liu_NB classifier =new Liu_NB();
		//通过训练集训练处一个分类模型
		classifier.buildclassifier(dataTrain);
		Liu_Performance eva=new Liu_Performance();
		eva.evalute(classifier, dataTest);
	}
	
	public static void main(String[] argv) throws Exception{
		String basedPath = "/Users/macdowell/Desktop/Professional Materials/3_Junior_S2/2_MachineLearning/Lab/3_NaiveBayes/";
		DataSource trainingSet=new DataSource( basedPath + "watermelon_train.arff");
		Instances instancesTrain = trainingSet.getDataSet();
		instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1); // 设置最后一列为属性类别
		
		DataSource testingSet = new DataSource( basedPath + "watermelon_test.arff");
		Instances instancesTest = testingSet.getDataSet();
		instancesTest.setClassIndex(instancesTest.numAttributes() - 1); // 设置最后一列为属性类别
		
		//对没有类标签的样例删除，减小噪音
		instancesTrain.deleteWithMissingClass();
		
		System.out.println("删除无标签样例 完成");
		//进行测试集与训练集评估
		evaluateNaiveBayes(instancesTrain,instancesTest);	
	}
}
