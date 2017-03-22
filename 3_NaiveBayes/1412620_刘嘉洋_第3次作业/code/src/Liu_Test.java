import java.util.Random;

import weka.classifiers.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 * MachineLearning Lab3: My NaiveBayes Classifier
 * 
 * @author LIU Jiayang. 1412620. NKCS
 * @date Thu 16 Mar 2017
 */
public class Liu_Test {
	public static void main(String[] args) throws Exception {
		String basedPath = "/Users/macdowell/Desktop/Professional Materials/3_Junior_S2/2_MachineLearning/Lab/3_NaiveBayes/";
		DataSource sourceTrain = new DataSource( basedPath + "watermelon_train.arff" );
		Instances dataTrain = sourceTrain.getDataSet();
		dataTrain.setClassIndex(dataTrain.numAttributes() - 1);
		Liu_NB my_NB = new Liu_NB();
		
		// 十折交叉验证
//		Evaluation evaluation = new Evaluation(dataTrain);
//		evaluation.crossValidateModel(my_NB, dataTrain, 10, new Random(1));
//		System.out.println(evaluation.pctCorrect());
//		System.out.println(evaluation.toSummaryString());

		// 测试集
		DataSource sourceTest = new DataSource(basedPath + "watermelon_test.arff");
		Instances dataTest = sourceTest.getDataSet();
		dataTest.setClassIndex(dataTest.numAttributes() - 1);
		Liu_NB cModel = new Liu_NB();
		Evaluation evaluation2 = new Evaluation(dataTrain);
		cModel.buildClassifier(dataTrain);
		evaluation2.evaluateModel(cModel, dataTest);
		System.out.println(evaluation2.pctCorrect());
		System.out.println(evaluation2.toSummaryString());
		
		// 拉普拉斯平滑测试
		DataSource sourceSmoothing = new DataSource(basedPath + "smoothing_test.arff");
		Instances dataSmoothing = sourceSmoothing.getDataSet();
		dataSmoothing.setClassIndex(dataSmoothing.numAttributes() - 1);
		Liu_NB dModel = new Liu_NB();
		Evaluation evaluation3 = new Evaluation(dataTrain);
		dModel.buildClassifier(dataTrain);
		evaluation3.evaluateModel(dModel, dataSmoothing);
		System.out.println("拉普拉斯平滑测试");
		System.out.println(evaluation3.toSummaryString());
	}
}
