import java.io.FileReader;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;

/**
 * MachineLearning Lab2, Assignment2: Filters
 * 
 * @author LIU Jiayang. 1412620. NKCS
 * @date Tue 7 Mar 2017
 */
public class FiltersAssign {
	public static Instances getFileInstances(String fileName) throws Exception {
		FileReader frData = new FileReader(fileName);
		Instances data = new Instances(frData);
		return data;
	}

	/**
	 * 属性离散化方法
	 * @param instances 数据实例
	 * @throws Exception
	 */
	public static void discretizeAttribute(Instances instances) throws Exception {
		// 离散化第1、4个属性
		Discretize discretize = new Discretize();
		String[] options = new String[2];
		options[0] = "-R";	// Specifies list of columns to Discretize.
		options[1] = "1,4"; // 指定多列 中间以,间隔
		// options[2] = "-R"; // 这样的写法是不对的
		// options[3] = "4";
		discretize.setOptions(options);
		discretize.setInputFormat(instances);
		Instances newData = Filter.useFilter(instances, discretize);
		for (int i = 0; i < 20; i++) {
			// instance( i )是得到第i个样本
			System.out.println(newData.instance(i));
		}
	}

	/**
	 * 属性归一化方法
	 * @param instances 数据实例
	 * @throws Exception
	 */
	public static void normalizeAttribute(Instances instances) throws Exception {
		Normalize normalize = new Normalize();
		String[] options = new String[4];
		// 归一化所有numeric类型的属性
		// 归一区间设置为[-1,1]
		options[0] = "-T"; // The translation of the output range.
		options[1] = "-1"; // 从-1起始
		options[2] = "-S"; // The scaling factor for the output range.
		options[3] = "2"; // 跨度为2
		normalize.setOptions(options);
		normalize.setInputFormat(instances);
		Instances newData = Filter.useFilter(instances, normalize);
		for (int i = 0; i < 20; i++) {
			System.out.println(newData.instance(i));
		}
	}

	/**
	 * 属性删除方法
	 * @param instances 数据实例
	 * @throws Exception
	 */
	public static void removeAttribute(Instances instances) throws Exception {
		Remove remove = new Remove();
		String[] options = new String[2];
		// 删除第2、3个属性
		options[0] = "-R";
		options[1] = "2,3"; // 指定多列 中间以,间隔
		remove.setOptions(options);
		remove.setInputFormat(instances);
		Instances newData = Filter.useFilter(instances, remove);
		for (int i = 0; i < 20; i++) {
			System.out.println(newData.instance(i));
		}
	}

	public static void main(String[] args) throws Exception {
		Instances instances = getFileInstances(
				"/Users/macdowell/Desktop/Professional Materials/3_Junior_S2/2_MachineLearning/Lab/2_WekaAPI/实验二 数据集/bank.arff");
		instances.setClassIndex(instances.numAttributes() - 1); // 设置最后一列为类别

		System.out.println("------------------离散化属性------------------");
		discretizeAttribute(instances);
		System.out.println("\n\n------------------归一化属性------------------");
		normalizeAttribute(instances);
		System.out.println("\n\n------------------删除属性------------------");
		removeAttribute(instances);
	}
}
