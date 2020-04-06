# 【ai】第二课-神经网络编程基础

### logistic回归

#### 损失函数

$$
-(ylog (a) +(1-y)log(1-a))
$$

这里的y*表示y帽

#### 成本函数

用于衡量w和b的效果

#### 向量化

z = np.dot(w,x) 是在计算w^Tx

向量化代码会提高代码运行速度

```python
import numpy as np
u = np.exp(v)
u = np.log(v)
u = np.Abs(v)	#计算绝对值
u = np.maximum(v)#计算所有元素中的最大值
v**2 #v中每个元素的平方
1/v  #就是每个元素求倒数

cal = a.sum(axis = 0)	#这个轴等于0意味着我希望python在竖直方向求和，水平为1
percentage = 100*a/(cal.reshape(1,4))	#让a矩阵除以一个1*4矩阵
```

python广播通用规则：如果你有m*n矩阵，然后+-x/一个1 x n矩阵python就会把他复制n次变成m x n 的矩阵然后逐项做运算，如果相反，用一个m x n 矩阵 +-乘除m x 1矩阵，南无也会复制n次，然后逐项运算。

#### 技巧

```python
import numpy as np
a = np.random.rand(5)		#生成5个随机高斯变量
print(a.shape)
#结果为(5,) 这是python秩为1的数组，不是行列向量
print(np.dot(a,a.T))
#不会得到一个矩阵，而是一个数字

#建议使用
a = np.random.randn(5,1)
#可以令a变成5x1列向量
a = a.reshape((5,1))
#使矩阵变成5x1列向量
```

