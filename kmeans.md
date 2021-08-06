#k-means 聚类算法实现实验报告

##实验原理：

####功能描述：
对于给定的样本集，按照样本之间的距离大小，将样本集划分为K个簇。让簇内的点尽量紧密的连在一起，而让簇间的距离尽量的大。

如果用数据表达式表示，假设簇划分为(C1,C2,...Ck)
，则我们的目标是最小化平方误差E：
$E=∑_{i=1}^k ∑_{x∈C_i}||x−μ_i||^2$

其中μi是簇Ci的均值向量，有时也称为质心，表达式为：
$μ_i = \frac{1}{|Ci|}∑_{x∈C_i}x$

####方法：
•1.初始化k个聚类中心𝝁𝒊
•2.将每个x划入到距离最近的聚类中心𝝁𝒊
•3.重新计算每个类的中心𝝁𝒊
•4.转至2继续执行，直至聚类不发生变化

##代码实现：

####准备：
使用引用鸢尾花卉数据集，通过PCA主成分表示前两维
代码：
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import random 

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
data = df.iloc[:, :4].to_numpy()


def pca(data, Decrease_dim):
    data = data.T
    mean = data.mean(axis = 1).reshape(-1, 1)
    data = data - mean   
    cov_matrix = 1 / len(data[0]) * np.matmul(data , data.T)

    lam,lam_vec = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(lam)[::-1]
    lam_new = lam[sorted_indices]
    new_data = lam_vec.T @ data
    new_data = new_data[sorted_indices,:]
    return lam_new,new_data

var,new_data = pca(data,2)
```

####自行实现k-means:

```python
#定义欧式距离
def distance(dot1,dot2):
    return np.sqrt(sum((dot1 - dot2)**2))

#初始化随机聚点
def randdot(data,k):
    n = np.shape(data)[1]
    dots = np.zeros((k,n))
    for i in range(0,n):
        min_doti = min(data[:,i])
        max_doti = max(data[:,i])
        rangi = max_doti - min_doti
        dots[:,i] = (min_doti + rangi * np.random.rand(k, 1)).reshape(-1)
    return dots

#实现k-means
def kmeans(data, k, dist = distance, createdot = randdot):
    
    #数据集的样本个数
    m = len(data)

    #每个样本点的包含两个数据，与聚点的距离与归属于哪个聚点
    dots_characteristic = np.zeros((m,2))
    
    #初始化生成聚点
    cdots = createdot(data,k)
    
    #判断聚点是否稳定
    if_change = True
    
    while if_change:
        if_change = False

        #找到每个点对应的最近聚点
        for  i in range(0,m):
            mindist = 1000000
            min_kind = -1
            for j in range(0,k):
                dis = distance(cdots[j,:],data[i,:])
                if dis < mindist:
                    min_kind = j
                    mindist = dis
            
            #修改每个点的数据
            if dots_characteristic[i,0] != min_kind:
                if_change = True
                dots_characteristic[i,0] = min_kind
                dots_characteristic[i,1] = mindist**2
            
            重新寻找新的聚点
            for i in range(0,k):
                same_kind_posi = data[[j for j in range(0,m) if dots_characteristic[j,0] == i]]
                cdots[i,:] = np.mean(same_kind_posi,axis = 0)
    return cdots,dots_characteristic

#set2保存有每个点所属聚类的数据，
#选择分别选择生成3个聚类与2个聚类观察图样特征
set1,set2 = kmeans(new_data.T,3)

#绘图
colour = []
for i in range(0,150):
    if set2[i,0] == 0:
        colour.append('red')
    elif set2[i,0] == 1:
        colour.append('yellow')
    else:
        colour.append('blue')
plt.scatter(x_new, y_new, c=colour)
```

生成图片：
![](images/2021-06-13-14-28-17.png)

经观察，无论选择生成两个聚类还是三个聚类，生成样本图都相同，原因可能是右侧的聚类较为紧凑，一种聚类的全部样本被归并到其他聚类。

####sklearn实现：

```python
x_new,y_new = new_data.T[:,0].reshape(-1),new_data.T[:,1].reshape(-1)

#分别尝试实现2个和3个聚类，将n_cluster分为2和3
model = KMeans(n_clusters=2)
model.fit(new_data.T)
#预测全部150条数据
all_predictions = model.predict(new_data.T)
 
#打印出来对150条数据的聚类散点图
plt.scatter(x_new, y_new, c=all_predictions)
plt.show()
```
分别尝试两个和三个聚类

图样：
![](images/2021-06-13-14-36-13.png)

![](images/2021-06-13-14-36-42.png)

说明sklearn在取聚点时会保持一定距离，使聚点不易消散