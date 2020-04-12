# dnn_app_utils_v2.py文件的函数笔记
实现一个神经网络需要的函数很多，其中很多函数是同一类型的，比如正向传播，反向传播。这些函数在神经网络模型中实现相似的功能，但是因为实现功能的要求的不同而不同。我在学习的过程中发现如果对神经网络的结构的把握不清晰，就很难写对相应的函数，因为功能相近的函数会有干扰。有练习次数少的问题，让我写一个神经网络很费力，我的下一个目标是不查资料，完全依靠自己对模型的理解写出一个可以运行的简单神经网络。
```python
def sigmoid(Z): # sigmoid函数
    A = 1 / (1 + np.exp(-Z))
    cache = Z
        
    return A, cache
   
def sigmoid_backward(dA, cache): # sigmoid反向传播函数
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert(dZ.shape == Z.shape) # 断言：如果dZ矩阵等于Z矩阵则报错
       
    return dZ
       
def relu(Z): # ReLU函数
    A = np.maximum(0, z)
    assert(A.shape == Z.shape)
    cache = Z
        
    return A, cache
        
def relu_backward(dA, cache): # ReLU反向传播函数
    Z = cache
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)
        
    return dZ
    
# 这是一个提取数据集的函数
def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # 你的训练集的特点
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # 你的训练集的标签

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # 你的测试集的特点
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # 你的测试集的标签

    classes = np.array(test_dataset["list_classes"][:]) # 类的列表
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# 初始化网络参数的函数
def initialize_parameters(n_x, n_h, n_y):
    """
    输入：
    n_x--输入层的大小
    n_h--隐藏层的大小
    n_y--输出层的大小
    返回：、
    参数--返回一个字典
    W1--形状的权值矩阵(n_h,n_x) 
    b1--形状偏置向量(n_h,1)
    W2--形状的权值矩阵(n_y,n_h) 
    b2--形状偏置向量(n_y,1)
    """
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x)*0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y, 1))
    
    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
    
# 深度初始化参数
def initialize_parameters_deep(layer_dims):
    '''
    输入:
    包含网络中每一层的维数的python数组(列表)
    
    返回:
    参数--python字典包含参数“W1”，“b1”，…
    Wl--形状的权值矩阵(layer_dims[l]， layer_dims[l-1])
    bl--形状的偏置向量(layer_dims[l]， 1)
    '''
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # 神经网络的层数

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters
    
# 线性正向传播函数
def linear_forward(A, W, b):
    '''
    实现一个层的正向传播的线性部分。
    输入:
    A--前一层(或输入数据)的激活:(size of previous layer, number of examples)
    W--权值矩阵:形状的numpy数组(size of current layer, size of previous layer)
    b--偏置向量，形状的numpy数组(size of the current layer, 1)
    返回:
    Z--激活函数的输入，也称预激活参数
    cache--一个包含“a”、“W”和“b”的python字典;为了有效地计算后向遍历而存储
    '''
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache
    
# 正向线性激活函数
def linear_activation_forward(A_prev, W, b, activation):
    '''
    实现线性->激活层的正向传播
    输入:
    A_prev--前一层(或输入数据)的激活:(前一层的大小，例数)
    W--权值矩阵:形状的numpy数组(当前层的大小，上一层的大小)
    b--偏置向量，形状的numpy数组(当前层的大小，1)
    激活--在这一层中使用的激活，以文本字符串的形式存储:“sigmoid”或“relu”
    返回:
    A--激活函数的输出，也称激活后值
    缓存--一个包含“linear_cache”和“activation_cache”的python字典;
    '''
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache
    
# 正向传播函数
def L_model_forward(X, parameters):
    '''
    实现对[输入层->ReLU函数]*(L-1)->隐藏层->sigmoid函数计算的正向传播
    输入:
    X--数据，形状的numpy数组(输入大小，实例数量)
    参数--initialize_parameters_deep()的输出
    返回:
    AL--最后的激活后值
    cache--缓存列表包含:
    linear_relu_forward()的每个缓存(有L-1个，索引从0到L-2)
    linear_sigmoid_forward()的缓存(有一个，索引为L-1)
    '''
    caches = []
    A = X
    L = len(parameters) // 2 # 神经网络的层数
    
    # 实现[LINEAR -> RELU]*(L-1)。将“cache”添加到“caches”列表中
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    # 实现线性-> SIGMOID。将“cache”添加到“caches”列表中。
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)

    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches
   
# 代价函数
def compute_cost(AL, Y):
    """
    输入:
    AL--与你的标签预测相对应的概率向量，形状(1, number of examples)
    Y--真正的“label”向量(例如:包含0 if non-cat, 1 if cat)， shape(1, number of examples)
    返回:
    cost--交叉熵成本
    """
    
    m = Y.shape[1]

    # 计算aL和y的损失。
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # 确保你的成本的形状是我们所期望的
    assert(cost.shape == ())
    
    return cost

# 反向传播函数
def linear_backward(dZ, cache):
    """
    实现单个层(层l)的反向传播的线性部分
    输入:
    dZ——成本相对于线性输出(当前层l)的梯度
    cache——值的元组(A_prev, W, b)来自当前层的正向传播
    返回:
    dA_prev——成本相对于激活(前一层l-1)的梯度，形状与A_prev相同
    dW——成本相对于W(当前层l)的梯度，形状与W相同
    db——成本相对于b(当前层l)的梯度，形状与b相同
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
# 反向传播函数
def linear_activation_backward(dA, cache, activation):
    """
    实现线性->激活层的反向传播。
    输入:
    dA——电流层l的激活后梯度
    cache——我们存储的值的元组(linear_cache、activation_cache)用于有效地计算反向传播
    activation——在这一层中使用的激活，以文本字符串的形式存储:“sigmoid”或“relu”
    
    返回:
    dA_prev——成本相对于激活(前一层l-1)的梯度，形状与A_prev相同
    dW——成本相对于W(当前层l)的梯度，形状与W相同
    db——成本相对于b(当前层l)的梯度，形状与b相同
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

# 反向传播函数
def L_model_backward(AL, Y, caches):
    """
    实现[LINEAR->RELU] * (L-1) -> LINEAR-> SIGMOID group的反向传播
    
    输入:
    AL --  概率向量，正向传播的输出(L_model_forward())
    Y -- 真正的“标签”向量(containing 0 if non-cat, 1 if cat)
    caches -- 包含缓存的列表:
    每个带有“relu”的linear_activation_forward()缓存(有(L-1)或它们，索引从0到L-2)
带有“sigmoid”的linear_activation_forward()的缓存 (there is one, index L-1)

    返回:
    grads -- 有斜度的字典
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    grads = {}
    L = len(caches) # 层数
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # 在这行之后，Y和AL的形状是一样的
    
    # 初始化反向传播
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # 第l层(SIGMOID ->线性)梯度。输入: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, activation = "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

# 更新参数的函数
def update_parameters(parameters, grads, learning_rate):
    """
    使用梯度下降更新参数
    输入:
    parameters -- 包含参数的字典
    grads -- python字典包含您的梯度，l_model_backwards的输出
    
    返回:
    parameters -- 包含更新参数的字典
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # 神经网络的层数

    # 更新每个参数的规则。使用for循环。
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

# 预测函数
def predict(X, y, parameters):
    """
    该函数用于预测l层神经网络的结果。
    参数:
    X -- 您希望标记的示例数据集
    参数 -- 训练模型的参数
    返回:
    p -- 对给定数据集X的预测
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # 神经网络的层数
    p = np.zeros((1,m))
    
    # 向前传播
    probas, caches = L_model_forward(X, parameters)

    
    # 将probas转换为0/1的标准化预测
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p

# 输出图像函数
def print_mislabeled_images(classes, X, y, p):
    """
    情节图像的预测和真相是不同的。
    X -- 数据集
    y -- 真标签
    p -- 预测
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # 设置图的默认大小
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))

```