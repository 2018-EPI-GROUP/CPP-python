# week2
这周学习了两层神经网络的搭建
```python
# 导入库
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from dnn_app_utils_v2 import * # 由于这个文件中函数比较多影响代码的感观，我把有关函数的部分写在另一篇笔记中

%matplotlib inline # 由于 %matplotlib inline 的存在，当输入plt.plot(x,y)后，不必再输入 plt.show()，图像将自动显示出来
plt.rcParams['figure.figsize'] = (5.0, 4.0) # 设置图的默认大小
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload # 在执行用户代码前，重新装入 软件的扩展和模块。
%autoreload 2 # autoreload 意思是自动重新装入,参数2：装入所有 %aimport 不包含的模块

np.random.seed(1) # 生成一个确定的随机数

# 初始化数据集
m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

# 平面化、向量化训练集和测试集
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T 
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# 标准化，使特征值处于0和1之间
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

# 定义模型的常数
n_x = 12288 
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)

# 二层模型函数
def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    '''
    实现了一个两层神经网络：输入层->ReLU->隐藏层->sigmoid
    输入：
    X -- 输入数据，形状(n_x，number of examples)
    Y -- 真正的“标签”向量(containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- 图层尺寸 (n_x, n_h, n_y)
    num_iterations -- 优化循环的迭代次数
    learning_rate -- 梯度下降更新规则的学习率
    print_cost -- 如果设置为True，这将每100次迭代打印成本
    
    返回：
    参数——一个包含W1、W2、b1和b2的字典
    '''
    np.random.seed(1)
    grads = {}
    costs = []
    m = X.shape[1] 
    (n_x, n_h, n_y) = layers_dims
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
# 梯度下降
    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")
        cost = compute_cost(A2, Y)
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation="sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation="relu")
        # 更新参数
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        parameters = update_parameters(parameters, grads, learning_rate)
        # 从参数中检索W1, b1, W2, b2
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        # 每100个训练例子打印成本
        if print_cost and i % 100 == 0:
            print("迭代后成本 {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    # 绘制图像
    plt.plot(np.squeeze(costs))
    plt.ylabel('成本')
    plt.xlabel('迭代次数 (每十)')
    plt.title("学习率 =" + str(learning_rate))
    plt.show()
    
    return parameters
    
# 训练模型并打印图像
parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
```