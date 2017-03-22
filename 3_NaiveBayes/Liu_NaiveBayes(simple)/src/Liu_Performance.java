import java.util.Arrays;

import weka.core.Instances;

/**
 * MachineLearning Lab3, Assignment 3_1: My Naive Bayes Classifier
 * 
 * @author LIU Jiayang. 1412620. NKCS
 * @date Tue 14 Mar 2017
 */
public class Liu_Performance {
	public void evalute(Liu_NB classifier,Instances instances)throws Exception{
 	   Instances testinstances=new Instances(instances);
 	   //遍历所有的实例
 	   int correct = 0,incorrect = 0;
		   for(int i=0;i<testinstances.numInstances();i++)
		   {
			  double probality = 0;
			  double compare[] = new double[testinstances.numInstances()];
			  //遍历所有最终分类的目标值
			  for(int k=0;k<Liu_NB.finalresultname.size();k++)
			  {
				  probality=0;
				//遍历所有的属性
				for(int j=0;j<testinstances.numAttributes()-1;j++)
				{
	   				double index;
	   				index=testinstances.instance(i).value(j);
					//遍历当前属性下的所有取值
					for(int r=0;r<Liu_NB.finalresultname.get(k).get(j).size();r++)
					{
						//找到当前属性对应的取值，计算概率
						if(index==Liu_NB.finalresultname.get(k).get(j).get(r).intValue())
						{
							if(probality==0)
								probality=probality+((double)Liu_NB.finalresultcount.get(k).get(j).get(r).intValue()/(double)Liu_NB.resultCount.get(k));
							else
								probality=probality*((double)Liu_NB.finalresultcount.get(k).get(j).get(r).intValue()/(double)Liu_NB.resultCount.get(k));
						}
					}
				}
				//将计算好的概率存入数组中
				compare[k]=probality*Liu_NB.aimProbability.get(k);
			  }
			 double temp[]=new double[testinstances.numInstances()];
			 for(int y=0;y<testinstances.numInstances();y++)
				 temp[y]=compare[y];
			 //将最大的类别调整至数组的最后
			 Arrays.sort(compare);
			 for(int l=0;l<testinstances.numInstances();l++)
			 {
				 //与最初的值进行比较，找出是否分类正确
				 if(temp[l]==compare[testinstances.numInstances()-1])
					 if(l==testinstances.instance(i).value(testinstances.instance(i).numAttributes()-1))
						 correct++;
					 else
					 {
						 incorrect++;
						 System.out.println(l);
					 }
			 }
		    }
			 System.out.println("正确分类比例：");
			 System.out.println((double)correct/(double)testinstances.numInstances());
			 System.out.println("错误分类比例：");
			 System.out.println((double)incorrect/(double)testinstances.numInstances());
		  
	}
}
