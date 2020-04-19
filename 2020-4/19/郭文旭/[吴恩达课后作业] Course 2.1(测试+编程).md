# [吴恩达课后作业] Course 2.1(测试+编程)

### 1.测试

1. 如果可测试样例非常多可以选择绝大部分训练，少量开发和测试
2. 开发集和测试集应该来自同一分布
3. 方差高可能是过拟合和欠拟合的情况，添加正则化、获取更多的训练数据都是解决方法
4. 开发集错误率大于训练集发生过拟合
5. 正则化技术导致梯度下降在每次迭代时权重收缩
6. 增加正则化超参数权重会变的非常小
7. 训练通过dropout来加强或减弱某部分的值，测试之中就不需要使用droput了
8. keep_prob增加会让正则化效应减弱，使得神经网络在结束时会比训练集表现好
9. Droput、L2正则化、扩充数据集都可以减小方差
10. 归一化输入x可以使得成本函数更快进行优化

### 2.编程

因为我们验证的初始化参数，正则化模型，和梯度校验设置的三层神经网络代码类似，只列出了初始化参数的网路部分

1. #### 初始化参数

   初始化参数设置的好，学习效率会比较高，最后的方差也比较小，图像效果很好

   ```python
   #!/usr/bin/env python
   # coding: utf-8
   
   import numpy as np
   import matplotlib.pyplot as plt
   import sklearn
   import sklearn.datasets
   plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
   plt.rcParams['image.interpolation'] = 'nearest'
   plt.rcParams['image.cmap'] = 'gray'
   
   train_X, train_Y, test_X, test_Y =load_dataset(is_plot=True)
   
   #调用三层神经网络的模型
   def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization="he",is_polt=True):
       """
       实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
       
       参数：
           X - 输入的数据，维度为(2, 要训练/测试的数量)
           Y - 标签，【0 | 1】，维度为(1，对应的是输入的数据的标签)
           learning_rate - 学习速率
           num_iterations - 迭代的次数
           print_cost - 是否打印成本值，每迭代1000次打印一次
           initialization - 字符串类型，初始化的类型【"zeros" | "random" | "he"】
           is_polt - 是否绘制梯度下降的曲线图
       返回
           parameters - 学习后的参数
       """
       grads = {}
       costs = []
       m = X.shape[1]
       layers_dims = [X.shape[0],10,5,1]
       
       #选择初始化参数的类型
       if initialization == "zeros":
           parameters = initialize_parameters_zeros(layers_dims)
       elif initialization == "random":
           parameters = initialize_parameters_random(layers_dims)
       elif initialization == "he":
           parameters = initialize_parameters_he(layers_dims)
       else : 
           print("错误的初始化参数！程序退出")
           exit
       
       #开始学习
       for i in range(0,num_iterations):
           #前向传播
           a3 , cache = forward_propagation(X,parameters)
           
           #计算成本        
           cost = compute_loss(a3,Y)
           
           #反向传播
           grads = backward_propagation(X,Y,cache)
           
           #更新参数
           parameters = update_parameters(parameters,grads,learning_rate)
           
           #记录成本
           if i % 1000 == 0:
               costs.append(cost)
               #打印成本
               if print_cost:
                   print("第" + str(i) + "次迭代，成本值为：" + str(cost))
           
    
       #学习完毕，绘制成本曲线
       if is_polt:
           plt.plot(costs)
           plt.ylabel('cost')
           plt.xlabel('iterations (per hundreds)')
           plt.title("Learning rate =" + str(learning_rate))
           plt.show()
       
       #返回学习完毕后的参数
       return parameters
   
   
   '''初始化为零效果非常差，学习率基本没有变化
       初始化为随机数效果会有不错的效果，但会逐渐便好
       抑梯度异常初始化，再三个初始化中效果最好
   '''
   def initialize_parameters_he(layers_dims):
       """
       参数：
           layers_dims - 列表，模型的层数和对应每一层的节点的数量
       返回
           parameters - 包含了所有W和b的字典
               W1 - 权重矩阵，维度为（layers_dims[1], layers_dims[0]）
               b1 - 偏置向量，维度为（layers_dims[1],1）
               ···
               WL - 权重矩阵，维度为（layers_dims[L], layers_dims[L -1]）
               b1 - 偏置向量，维度为（layers_dims[L],1）
       """
       
       np.random.seed(3)               # 指定随机种子
       parameters = {}
       L = len(layers_dims)            # 层数
       
       for l in range(1, L):
           parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l - 1])
           parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
           
           #使用断言确保我的数据格式是正确的
           assert(parameters["W" + str(l)].shape == (layers_dims[l],layers_dims[l-1]))
           assert(parameters["b" + str(l)].shape == (layers_dims[l],1))
           
       return parameters
   
   
   # In[32]:
   
   
   parameters = model(train_X, train_Y, initialization = "he",is_polt=True)
   print("训练集:")
   predictions_train = predict(train_X, train_Y, parameters)
   print("测试集:")
   predictions_test = predict(test_X, test_Y, parameters)
   
   
   # In[27]:
   
   
   #构造一个三层的神经网络，以及生成一个圆形数据以及可视化函数
   # -*- coding: utf-8 -*-
   
   #init_utils.py
   
   import numpy as np
   import matplotlib.pyplot as plt
   import sklearn
   import sklearn.datasets
   
   
   def sigmoid(x):
       """
       Compute the sigmoid of x
    
       Arguments:
       x -- A scalar or numpy array of any size.
    
       Return:
       s -- sigmoid(x)
       """
       s = 1/(1+np.exp(-x))
       return s
    
   def relu(x):
       """
       Compute the relu of x
    
       Arguments:
       x -- A scalar or numpy array of any size.
    
       Return:
       s -- relu(x)
       """
       s = np.maximum(0,x)
       
       return s
       
   def compute_loss(a3, Y):
       
       """
       Implement the loss function
       
       Arguments:
       a3 -- post-activation, output of forward propagation
       Y -- "true" labels vector, same shape as a3
       
       Returns:
       loss - value of the loss function
       """
       
       m = Y.shape[1]
       logprobs = np.multiply(-np.log(a3),Y) + np.multiply(-np.log(1 - a3), 1 - Y)
       loss = 1./m * np.nansum(logprobs)
       
       return loss
       
   def forward_propagation(X, parameters):
       """
       Implements the forward propagation (and computes the loss) presented in Figure 2.
       
       Arguments:
       X -- input dataset, of shape (input size, number of examples)
       Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
       parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                       W1 -- weight matrix of shape ()
                       b1 -- bias vector of shape ()
                       W2 -- weight matrix of shape ()
                       b2 -- bias vector of shape ()
                       W3 -- weight matrix of shape ()
                       b3 -- bias vector of shape ()
       
       Returns:
       loss -- the loss function (vanilla logistic loss)
       """
           
       # retrieve parameters
       W1 = parameters["W1"]
       b1 = parameters["b1"]
       W2 = parameters["W2"]
       b2 = parameters["b2"]
       W3 = parameters["W3"]
       b3 = parameters["b3"]
       
       # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
       z1 = np.dot(W1, X) + b1
       a1 = relu(z1)
       z2 = np.dot(W2, a1) + b2
       a2 = relu(z2)
       z3 = np.dot(W3, a2) + b3
       a3 = sigmoid(z3)
       
       cache = (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3)
       
       return a3, cache
    
   def backward_propagation(X, Y, cache):
       """
       Implement the backward propagation presented in figure 2.
       
       Arguments:
       X -- input dataset, of shape (input size, number of examples)
       Y -- true "label" vector (containing 0 if cat, 1 if non-cat)
       cache -- cache output from forward_propagation()
       
       Returns:
       gradients -- A dictionary with the gradients with respect to each parameter, activation and pre-activation variables
       """
       m = X.shape[1]
       (z1, a1, W1, b1, z2, a2, W2, b2, z3, a3, W3, b3) = cache
       
       dz3 = 1./m * (a3 - Y)
       dW3 = np.dot(dz3, a2.T)
       db3 = np.sum(dz3, axis=1, keepdims = True)
       
       da2 = np.dot(W3.T, dz3)
       dz2 = np.multiply(da2, np.int64(a2 > 0))
       dW2 = np.dot(dz2, a1.T)
       db2 = np.sum(dz2, axis=1, keepdims = True)
       
       da1 = np.dot(W2.T, dz2)
       dz1 = np.multiply(da1, np.int64(a1 > 0))
       dW1 = np.dot(dz1, X.T)
       db1 = np.sum(dz1, axis=1, keepdims = True)
       
       gradients = {"dz3": dz3, "dW3": dW3, "db3": db3,
                    "da2": da2, "dz2": dz2, "dW2": dW2, "db2": db2,
                    "da1": da1, "dz1": dz1, "dW1": dW1, "db1": db1}
       
       return gradients
    
   def update_parameters(parameters, grads, learning_rate):
       """
       Update parameters using gradient descent
       
       Arguments:
       parameters -- python dictionary containing your parameters 
       grads -- python dictionary containing your gradients, output of n_model_backward
       
       Returns:
       parameters -- python dictionary containing your updated parameters 
                     parameters['W' + str(i)] = ... 
                     parameters['b' + str(i)] = ...
       """
       
       L = len(parameters) // 2 # number of layers in the neural networks
    
       # Update rule for each parameter
       for k in range(L):
           parameters["W" + str(k+1)] = parameters["W" + str(k+1)] - learning_rate * grads["dW" + str(k+1)]
           parameters["b" + str(k+1)] = parameters["b" + str(k+1)] - learning_rate * grads["db" + str(k+1)]
           
       return parameters
       
   def predict(X, y, parameters):
       """
       This function is used to predict the results of a  n-layer neural network.
       
       Arguments:
       X -- data set of examples you would like to label
       parameters -- parameters of the trained model
       
       Returns:
       p -- predictions for the given dataset X
       """
       
       m = X.shape[1]
       p = np.zeros((1,m), dtype = np.int)
       
       # Forward propagation
       a3, caches = forward_propagation(X, parameters)
       
       # convert probas to 0/1 predictions
       for i in range(0, a3.shape[1]):
           if a3[0,i] > 0.5:
               p[0,i] = 1
           else:
               p[0,i] = 0
    
       # print results
       print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
       
       return p
       
   def load_dataset(is_plot=True):
       np.random.seed(1)
       train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
       np.random.seed(2)
       test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
       # Visualize the data
       if is_plot:
           plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
       train_X = train_X.T
       train_Y = train_Y.reshape((1, train_Y.shape[0]))
       test_X = test_X.T
       test_Y = test_Y.reshape((1, test_Y.shape[0]))
       return train_X, train_Y, test_X, test_Y
    
   def plot_decision_boundary(model, X, y):
       # Set min and max values and give it some padding
       x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
       y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
       h = 0.01
       # Generate a grid of points with distance h between them
       xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
       # Predict the function value for the whole grid
       Z = model(np.c_[xx.ravel(), yy.ravel()])
       Z = Z.reshape(xx.shape)
       # Plot the contour and training examples
       plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
       plt.ylabel('x2')
       plt.xlabel('x1')
       plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
       plt.show()
    
   def predict_dec(parameters, X):
       """
       Used for plotting decision boundary.
       
       Arguments:
       parameters -- python dictionary containing your parameters 
       X -- input data of size (m, K)
       
       Returns
       predictions -- vector of predictions of our model (red: 0 / blue: 1)
       """
       
       # Predict using forward propagation and a classification threshold of 0.5
       a3, cache = forward_propagation(X, parameters)
       predictions = (a3>0.5)
       return predictions
   
   ```

2. #### 正则化

   在正则化的时候遇上了一些麻烦，因为作业中给的代码某些函数不符合新的anaconda标准，很难发现错误，最后百度发现了问题所在，修改之后可以顺利实现验证

   在验证中就可以发现，不使用正则化会有明显的过拟合特性

   下面列出使用正则化的代码

   **L2正则化**：避免过度拟合的标准方法，他是通过适当的修改成本函数。

   ```python
   def compute_cost_with_regularization(A3,Y,parameters,lambd):
       """
       实现公式2的L2正则化计算成本
       
       参数：
           A3 - 正向传播的输出结果，维度为（输出节点数量，训练/测试的数量）
           Y - 标签向量，与数据一一对应，维度为(输出节点数量，训练/测试的数量)
           parameters - 包含模型学习后的参数的字典
       返回：
           cost - 使用公式2计算出来的正则化损失的值
       
       """
       m = Y.shape[1]
       W1 = parameters["W1"]
       W2 = parameters["W2"]
       W3 = parameters["W3"]
       
       cross_entropy_cost = reg_utils.compute_cost(A3,Y)
       
       L2_regularization_cost = lambd * (np.sum(np.square(W1)) + np.sum(np.square(W2))  + np.sum(np.square(W3))) / (2 * m)
       
       cost = cross_entropy_cost + L2_regularization_cost
       
       return cost
   
   #当然，因为改变了成本函数，我们也必须改变向后传播的函数， 所有的梯度都必须根据这个新的成本值来计算。
   
   def backward_propagation_with_regularization(X, Y, cache, lambd):
       """
       实现我们添加了L2正则化的模型的后向传播。
       
       参数：
           X - 输入数据集，维度为（输入节点数量，数据集里面的数量）
           Y - 标签，维度为（输出节点数量，数据集里面的数量）
           cache - 来自forward_propagation（）的cache输出
           lambda - regularization超参数，实数
       
       返回：
           gradients - 一个包含了每个参数、激活值和预激活值变量的梯度的字典
       """
       
       m = X.shape[1]
       
       (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
       
       dZ3 = A3 - Y
       
       dW3 = (1 / m) * np.dot(dZ3,A2.T) + ((lambd * W3) / m )
       db3 = (1 / m) * np.sum(dZ3,axis=1,keepdims=True)
       
       dA2 = np.dot(W3.T,dZ3)
       dZ2 = np.multiply(dA2,np.int64(A2 > 0))
       dW2 = (1 / m) * np.dot(dZ2,A1.T) + ((lambd * W2) / m)
       db2 = (1 / m) * np.sum(dZ2,axis=1,keepdims=True)
       
       dA1 = np.dot(W2.T,dZ2)
       dZ1 = np.multiply(dA1,np.int64(A1 > 0))
       dW1 = (1 / m) * np.dot(dZ1,X.T) + ((lambd * W1) / m)
       db1 = (1 / m) * np.sum(dZ1,axis=1,keepdims=True)
       
       gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                    "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                    "dZ1": dZ1, "dW1": dW1, "db1": db1}
       
       return gradients
       
   
   ```

   打印效果

   ```python
   parameters = model(train_X, train_Y, lambd=0.7,is_plot=True)
   print("使用正则化，训练集:")
   predictions_train = predict(train_X, train_Y, parameters)
   print("使用正则化，测试集:")
   predictions_test = predict(test_X, test_Y, parameters)
   ```

   第0次迭代，成本值为：0.6974484493131264
   第10000次迭代，成本值为：0.2684918873282239
   第20000次迭代，成本值为：0.2680916337127301

   效果很好

    λ的值是可以使用开发集调整时的超参数。L2正则化会使决策边界更加平滑。如果λ太大，也可能会“过度平滑”，从而导致模型高偏差。L2正则化实际上在做什么？L2正则化依赖于较小权重的模型比具有较大权重的模型更简单这样的假设，因此，通过削减成本函数中权重的平方值，可以将所有权重值逐渐改变到到较小的值。权值数值高的话会有更平滑的模型，其中输入变化时输出变化更慢，但是你需要花费更多的时间。L2正则化对以下内容有影响：

   1. 成本计算       ： 正则化的计算需要添加到成本函数中
   2. 反向传播功能     ：在权重矩阵方面，梯度计算时也要依据正则化来做出相应的计算
   3. 重量变小（“重量衰减”) ：权重被逐渐改变到较小的值。
      

   **Drapout**：Dropout的原理就是每次迭代过程中随机将其中的一些节点失效。

   

   #### 3.梯度校验

   $$
   difference = \frac{||grad - gradapprox||_2}{||gard||_2+||gradapprox||_2}
   $$

   
