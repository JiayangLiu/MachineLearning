import java.util.ArrayList;
import java.util.Enumeration;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * MachineLearning Lab3, Assignment 3_1: My Naive Bayes Classifier
 * 
 * @author LIU Jiayang. 1412620. NKCS
 * @date Tue 14 Mar 2017
 */
public class Liu_NB {
	static ArrayList<ArrayList<ArrayList<Double>>> finalresultcount = new ArrayList<ArrayList<ArrayList<Double>>>();
	static ArrayList<ArrayList<ArrayList<Double>>> finalresultname = new ArrayList<ArrayList<ArrayList<Double>>>();
	static ArrayList<Float> aimProbability = new ArrayList<Float>();
	static ArrayList<Integer> resultCount = new ArrayList<Integer>();

	public static void buildclassifier(Instances instances) throws Exception {
		Instances m_instances = new Instances(instances);

		// 删除没有类标签的样例，减小噪音
		m_instances.deleteWithMissingClass();

		// 数据的实例个数
		int num_instances = m_instances.numInstances();
		System.out.println("数据的实例个数: " + num_instances);
		// 数据的属性的个数
		int num_attributes = m_instances.numAttributes();
		System.out.println("数据的属性个数: " + num_attributes);

		// 存储目标值的种类
		ArrayList<Double> result = new ArrayList<Double>();
		// 存储目标值的个数
		// ArrayList<Integer>resultCount=new ArrayList<Integer>();
		// 存储对应的label的概率
		// ArrayList<Float>aimProbability=new ArrayList<Float>();

		// 将数据集中的每一个实例按照各种属性存入数组中
		ArrayList<ArrayList<ArrayList<Double>>> ClassListBasedLabel = new ArrayList<ArrayList<ArrayList<Double>>>();

		// 计算P(Ci)的值
		// 遍历数据集，找出目标属性并计数
		for (int i = 0; i < num_instances; i++) {
			int j = 0;
			Instance currentInstance = m_instances.instance(i);
			// 返回最后一个属性的值
			double currentVal = currentInstance.value(num_attributes - 1);
			if (!result.contains(currentVal)) {
				result.add(j, currentVal);
				resultCount.add(j++, 1);
			} else {
				int curvalindex = result.indexOf(currentVal);
				int count = resultCount.get(curvalindex);
				resultCount.set(curvalindex, ++count);
			}
		}

		for (int k = 0; k < result.size(); k++) {
			double probabiblity;
			// 存储P(Yi）的概率
			aimProbability.add(k, (float) (resultCount.get(k) / (double) num_instances));
		}
		////////////////////////////////////////////////////////////////////////////////////////////////
		// 遍历各个属性

		// ArrayList<ArrayList<ArrayList<Double>>> finalresultcount=new
		// ArrayList<ArrayList<ArrayList<Double>>>();
		// ArrayList<ArrayList<ArrayList<Double>>> finalresultname=new
		// ArrayList<ArrayList<ArrayList<Double>>>();

		// 记录当前所判断的属性
		int attributecode = 0;
		for (int e = 0; e < num_attributes - 1; e++) {
			int sta = 0;
			int count = 1;
			for (int j = 0; j < result.size(); j++) {
				sta = 0;
				ArrayList<Double> featuresnames = new ArrayList<Double>();
				ArrayList<Double> featurescounts = new ArrayList<Double>();
				Enumeration enu = m_instances.enumerateAttributes();
				Attribute attribute = (Attribute) enu.nextElement();
				for (int m = 0; m < num_instances; m++) {
					Instance tempinstance = m_instances.instance(m);
					if (tempinstance.value(num_attributes - 1) == result.get(j)) {
						// 将取出实例的第attributecode个属性的值加入sample
						ArrayList<Double> sample = new ArrayList<Double>();
						sample.add(tempinstance.value(attributecode));
						// 判断name里面是否含有该属性值
						if (sta == 0) {
							// 第一次存放属性
							featuresnames.add(sta, tempinstance.value(attributecode));
							featurescounts.add(sta++, (double) count);
						} else {
							if (!featuresnames.contains(sample.get(0))) {
								// 不存在该属性
								featuresnames.add(sta, tempinstance.value(attributecode));
								featurescounts.add(sta++, (double) count);
							} else {
								// 存在该属性
								int subindex = featuresnames.indexOf(sample.get(0));
								Double tempcount = featurescounts.get(subindex);
								featurescounts.set(subindex, ++tempcount);
							}
						}
					}
				}

				if (attributecode == 0) {
					ArrayList<ArrayList<Double>> change = new ArrayList<ArrayList<Double>>();
					change.add(featurescounts);
					finalresultcount.add(j, change);
					ArrayList<ArrayList<Double>> change1 = new ArrayList<ArrayList<Double>>();
					change1.add(featuresnames);
					finalresultname.add(j, change1);
				} else {
					finalresultcount.get(j).add(attributecode, featurescounts);
					finalresultname.get(j).add(attributecode, featuresnames);
				}
			}
			attributecode++;
		}

	}
	
}
