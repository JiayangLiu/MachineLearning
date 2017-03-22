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
		System.out.println(eval.toClassDetailsString()); //���TP/TF�ʣ���׼�ʣ���ȫ�ʣ�Fֵ��ROC
		System.out.println( "***********************************\n\n" ); 
		System.out.println(eval.toSummaryString()); //�������ͳ������
		System.out.println( "***********************************\n\n" ); 
		System.out.println(eval.toMatrixString());//��������
		System.out.println( "***********************************\n\n" ); 
		System.out.println(eval.errorRate()); //������
		System.out.println( "***********************************\n\n" ); 
		System.out.println(eval.correct()); //��ȷ��������
		System.out.println( "***********************************\n\n" ); 
	}
	

	public static void main( String[] args ) throws Exception 
	{ 
		Instances insts = getFileInstances("C:/Users/Administrator/Desktop/ML��ѧʵϰ/2/bank.arff");
		crossValidation(insts); 
	}
}
