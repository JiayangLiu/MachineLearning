import java.util.Enumeration;

import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Capabilities.Capability;
import weka.core.Instances;

/**
 * MachineLearning Lab3: My NaiveBayes Classifier
 * 
 * @author LIU Jiayang. 1412620. NKCS
 * @date Thu 16 Mar 2017
 */
// Weka中的NaiveBayes便是继承了AbstraceClassifier类继承该类需要实现buildClassifier方法
public class Liu_NB extends AbstractClassifier {
	// 类别估计器 记录P(Y)的值
	protected Double[] m_classEstimator;
	// 类别数量 即instance.numclass()的返回值
	protected int class_num;
	// 保存和备份训练集
	protected Instances m_data;
	protected Integer[] times;
	// 存放各个P(X|Y)
	protected Liu_Estimator[][] m_Estimators;

	// 针对已知的训练集得出P(Y)和P(X|Y)
	@Override
	public void buildClassifier(Instances data) throws Exception {
		// 判断分类器能否处理该数据
		getCapabilities().testWithFail(data);
		m_data = new Instances(data);
		// 删除类型缺失的实例
		data.deleteWithMissingClass();
		// 类别数量
		class_num = data.numClasses();
		m_Estimators = new Liu_Estimator[m_data.numAttributes() - 1][m_data.numClasses()];

		for (int i = 0; i < m_data.numAttributes() - 1; i++)
			for (int j = 0; j < m_data.numClasses(); j++) {
				Liu_Estimator temp = new Liu_Estimator();
				m_Estimators[i][j] = temp;
			}

		// 创建属性分类器，计算P(X|Y)
		m_classEstimator = new Double[m_data.numClasses()];
		// 计算P(Y)
		// times数组用来记录每个y出现的次数
		times = new Integer[m_data.numClasses()];

		// 使用Laplace平滑
		int sum = 0;
		for (int i = 0; i < m_data.numClasses(); i++) {
			times[i] = 1;
			sum++;
		}
		
		// 遍历训练集 找出其每个的类别 统计 加到times中
		Enumeration enumInsts = data.enumerateInstances();
		while (enumInsts.hasMoreElements()) {
			Instance instance = (Instance) enumInsts.nextElement();
			Double now_label = instance.value(instance.classIndex());
			int now_class = now_label.intValue();
			if (now_class < m_data.numClasses()) {
				times[now_class]++;
				sum++;
			}
		}
		// 通过times计算得出对应的P(Y)
		for (int i = 0; i < m_data.numClasses(); i++) {
			m_classEstimator[i] = (double) (times[i] * (1.0) / sum);
		}

		// 下面开始计算各个P(X|Y)
		enumInsts = data.enumerateAttributes();
		int now = -1;
		// 遍历每一个属性 针对每一个属性的类型不同 分别对其m_Estimators做不同的初始化处理
		while (enumInsts.hasMoreElements()) {
			Attribute attribute = (Attribute) enumInsts.nextElement();
			// 对离散型的初始化处理
			if (attribute.type() == Attribute.NOMINAL) {
				if (attribute.index() == m_data.classIndex())
					continue;
				now++;
				// 获得当前属性下的具体数值
				int number = attribute.numValues();
				for (int i = 0; i < m_data.numClasses(); i++) {
					m_Estimators[now][i].init(number);
				}
				continue;
			}
			// 对连续型的初始化处理
			if (attribute.type() == Attribute.NUMERIC) {
				if (attribute.index() == m_data.classIndex())
					continue;
				now++;
				int number = attribute.numValues();
				for (int i = 0; i < m_data.numClasses(); i++) {
					m_Estimators[now][i].init(true);
				}
				continue;
			}
		}
		
		enumInsts = m_data.enumerateAttributes();
		// 对训练集每个取值的统计 根据属性类别的不同采用不同的处理方法
		while (enumInsts.hasMoreElements()) {
			Attribute attribute = (Attribute) enumInsts.nextElement();
			int number = attribute.index();
			if (number == m_data.classIndex())
				continue;
			
			// 对每个属性判断其类型进行处理
			if (attribute.type() == Attribute.NOMINAL) {
				Enumeration enumeration = data.enumerateInstances();
				while (enumeration.hasMoreElements()) {
					// 二维循环 遍历每一个数据下的每一个属性的值
					Instance instance = (Instance) enumeration.nextElement();
					Double now_tDouble = instance.value(attribute);
					int now_number = now_tDouble.intValue();
					int now_class_number = (int) (instance.value(instance.classIndex()));
					// 直接插入到对应的三维数组中
					m_Estimators[number][now_class_number].insert(now_number);
				}
				continue;
			}
			if (attribute.type() == Attribute.NUMERIC) {
				Enumeration enumeration = data.enumerateInstances();
				while (enumeration.hasMoreElements()) {
					Instance instance = (Instance) enumeration.nextElement();
					Double now_Double = instance.value(attribute);
					int now_class_number = (int) (instance.value(instance.classIndex()));
					// 对于连续性的变量 采用的方法是 一个一个插入 先拟合均值
					m_Estimators[number][now_class_number].insert_mu(now_Double);
				}
				// 对于每一个变量 遍历完成后 得出每一个均值
				for (int i = 0; i < m_data.numClasses(); i++) {
					m_Estimators[number][i].set_mu();
				}
				// 再次遍历整个训练集 根据刚拟合出的均值 再继续拟合方差
				enumeration = data.enumerateInstances();
				while (enumeration.hasMoreElements()) {
					Instance instance = (Instance) enumeration.nextElement();
					Double now_Double = instance.value(attribute);
					int now_class_number = (int) (instance.value(instance.classIndex()));
					m_Estimators[number][now_class_number].insert_simga(now_Double);
				}
			}
		}

		enumInsts = m_data.enumerateAttributes();
		while (enumInsts.hasMoreElements()) {
			Attribute attribute = (Attribute) enumInsts.nextElement();
			int number = attribute.index();
			if (number == m_data.classIndex())
				continue;
			// 针对离散型 全部遍历之后 生成每一个的概率
			if (attribute.type() == Attribute.NOMINAL) {
				for (int i = 0; i < m_data.numClasses(); i++) {
					m_Estimators[number][i].set_prob();
				}
			}
			// 针对连续型 全部遍历之后 拟合出每一个的方差
			if (attribute.type() == Attribute.NUMERIC) {
				for (int i = 0; i < m_data.numClasses(); i++) {
					m_Estimators[number][i].set_simga();
				}
			}
		}
	}

	// 用来确定该分类器可以处理的属性和类别的类型(主要是离散和连续)
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
		// 属性
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// 类别
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.NUMERIC_CLASS);

		// 实例 设置最少实例数量为0 即该算法能处理没有实例的数据
		result.setMinimumNumberInstances(0);

		return result;
	}

	// 对于一个特定的未知类别的数据 对其进行分类 返回值为一个double数组 表示每种取值的概率相对大小 接下来找出prob中最大的数作为该数据的类别
	public double[] distributionForInstance(Instance instance) {
		double[] prob = new double[m_data.numClasses()];
		// 先把每一个类别的P(Y)赋值给prob作为初始值
		for (int i = 0; i < m_data.numClasses(); i++) {
			prob[i] = m_classEstimator[i];
		}
		double[] temp;
		Enumeration enumeration = m_data.enumerateAttributes();
		double max = 0;
		// 遍历当前数据的所有属性
		while (enumeration.hasMoreElements()) {
			max = 0;
			Attribute attribute = (Attribute) enumeration.nextElement();
			int number = attribute.index();
			if (number == m_data.classIndex())
				continue;
			temp = new double[m_data.numClasses()];
			// 对于离散型属性 取得对应取值的各个类别的概率
			if (attribute.type() == Attribute.NOMINAL) {

				for (int i = 0; i < m_data.numClasses(); i++) {
					int now_number = (int) instance.value(attribute);
					temp[i] = m_Estimators[number][i].get_prob(now_number);
				}
			}
			// 对于连续型属性 直接按照所得到的高斯分布函数得出对应的概率
			if (attribute.type() == Attribute.NUMERIC) {
				for (int i = 0; i < m_data.numClasses(); i++) {
					double now_number = instance.value(attribute);
					temp[i] = Math.max(1e-75, m_Estimators[number][i].get_prob(now_number));
				}
			}
			for (int i = 0; i < m_data.numClasses(); i++) {
				prob[i] = prob[i] * temp[i];
				if (max < prob[i])
					max = prob[i];
			}
			// 防止下溢出处理
			if ((max > 0) && (max < 1e-75)) {
				for (int i = 0; i < m_data.numClasses(); i++)
					prob[i] = prob[i] * 1e75;
			}
		}
		return prob;
	}
}
