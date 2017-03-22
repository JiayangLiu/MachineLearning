import java.io.FileReader;

import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;


public class Test1 {
	public static Instances getFileInstances( String fileName ) throws Exception 
	{ 
		FileReader frData = new FileReader( fileName ); 
		Instances data = new Instances( frData ); 
		return data; 
	} 
	
	public static void discretizeAttribute(Instances instances) throws Exception{
		//***����һ��������ɢ��
		Discretize discretize=new Discretize();
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = "1";
		discretize.setOptions(options);
		discretize.setInputFormat(instances);
		Instances newData = Filter.useFilter(instances, discretize);
		for( int i = 0; i < 5; i++ ) 
		{ 
			//instance( i )�ǵõ���i������ 
			System.out.println( newData.instance( i ) ); 
		}
	}
	
	public static void main(String[] args) throws Exception 
	{ 
		Instances instances = getFileInstances("C:/Users/Administrator/Desktop/ML��ѧʵϰ/2/bank.arff"); 
		instances.setClassIndex( instances.numAttributes() - 1 );   //����ĳһ��Ϊ���

		
		discretizeAttribute(instances);  
	

	}
}
