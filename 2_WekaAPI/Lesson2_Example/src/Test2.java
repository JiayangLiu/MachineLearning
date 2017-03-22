import java.io.FileReader;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Debug.Random;
import weka.core.Instances;


public class Test2 {
	public static Instances getFileInstances( String fileName ) throws Exception
	{ 
		FileReader frData = new FileReader( fileName ); 
		Instances m_instances = new Instances( frData ); 

//		System.out.println(m_instances.);
		m_instances.setClassIndex( m_instances.numAttributes() - 1 ); 
		
		return m_instances;
	}
	
	public static void crossValidation(Instances m_instances) throws Exception 
	{ 
		J48 classifier = new J48();
			
		Evaluation eval = new Evaluation( m_instances ); 
		eval.crossValidateModel( classifier, m_instances, 10, new Random(1)); 
		System.out.println(eval.toClassDetailsString()); //输出TP/TF率，查准率，查全率，F值，ROC
		System.out.println( "***********************************\n\n" ); 
		System.out.println(eval.toSummaryString()); //输出性能统计数据
		System.out.println( "***********************************\n\n" ); 
		System.out.println(eval.toMatrixString());//混淆矩阵
		System.out.println( "***********************************\n\n" ); 
		System.out.println(eval.errorRate()); //错误率
		System.out.println( "***********************************\n\n" ); 
		System.out.println(eval.correct()); //正确样本个数
		System.out.println( "***********************************\n\n" ); 
	}
	

	public static void main( String[] args ) throws Exception 
	{ 
		Instances insts = getFileInstances("C:/Users/Administrator/Desktop/ML教学实习/2/bank.arff");
		crossValidation(insts); 
	}
}
