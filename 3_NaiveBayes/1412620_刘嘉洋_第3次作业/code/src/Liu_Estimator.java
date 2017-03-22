/**
 * MachineLearning Lab3: My NaiveBayes Classifier
 * 
 * @author LIU Jiayang. 1412620. NKCS
 * @date Thu 16 Mar 2017
 */
public class Liu_Estimator {
	public Liu_Estimator(int num) {
	}
	public Liu_Estimator() {
	}
	
	// ---离散型----------------------------------------------
	// 针对当前即第i个属性 第j个分类下 当前属性的各个取值的出现次数
	int[] times;
	// 根据times取得每个取值概率
	Double[] prob;
	// 各个取值出现次数的总和 用于统计和生成概率
	int sum;
	
	// 初始化函数
	public void init(int num) {
		flag = false;
		times = new int[num];
		prob = new Double[num];
		sum++;
		// 使用Laplace平滑
		for (int i = 0; i < num; i++) {
			times[i] = 1;
			sum++;
		}
		this.num = num;
	}

	// 插入函数
	public void insert(int num) {
		// 每次插入之后 总出现次数自增 对应的times值自增
		sum++;
		times[num]++;
	}
	public void insert(int num, int time) {
	    sum = sum + time;
	    times[num] = times[num] + time;
  }
	
	// 统计函数 在所有统计完成之后计算对应的probs
	public void set_prob() {
		for (int i = 0; i < num; i++) {
			prob[i] = (times[i] * 1.0) / sum;
		}
	}
	
	// ---连续型----------------------------------------------
	int num;
	double mu;
	double simga;
	int simga_sum;
	boolean flag;

	// 初始化函数
	public void init(boolean set) {
		sum = 0;
		mu = 0;
		simga = 0;
		simga_sum = 0;
		flag = true;
	}
	
	// 插入函数
	public void insert_mu(double x) {
		mu = mu + x;
		sum++;
	}
	public void insert_simga(double x) {
		simga = simga + (x - mu) * (x - mu);
		simga_sum++;
	}

	// 拟合函数 在对应的值全部累加完毕后 求出其值
	public void set_mu() {
		mu = (mu) / (sum * 1.0);
	}
	public void set_simga() {
		simga = (simga) / (simga_sum * 1.0);
	}

	public Double[] get_prob() {
		return prob;
	}
	public Double get_prob(int x) {
		if (x < num)
			return prob[x];
		return 0.0;
	}
	// 高斯分布函数 对于未知的数值x 给出其经过拟合后的概率 用于测试集
	public double get_prob(double x) {
		double ans = (((1.0) / (Math.sqrt((2 * Math.PI) * simga))) * Math.exp(-(((x - mu) * (x - mu)) / (2 * simga * simga))));
		return ans;
	}
}
