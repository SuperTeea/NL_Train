import numpy as np

def load_mnist():
    # 加载数据并归一化
    with np.load('Data/mnist.npz') as data:
        # 训练集（前50000） + 验证集（后10000） + 测试集（10000）
        X_train = data['x_train'][:50000].astype(np.float32) / 255.0  # 归一化并转float32
        Y_train = data['y_train'][:50000].astype(np.uint8)
        X_dev = data['x_train'][50000:].astype(np.float32) / 255.0
        Y_dev = data['y_train'][50000:].astype(np.uint8)
        X_test = data['x_test'].astype(np.float32) / 255.0
        Y_test = data['y_test'].astype(np.uint8)
        
        X_train = X_train.reshape(-1, 784, 1)
        X_dev = X_dev.reshape(-1, 784, 1)
        X_test = X_test.reshape(-1, 784, 1)
        
        return (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test)

def datas():
    (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test) = load_mnist()
    return (
        list(zip(X_train, [vectorized_result(y) for y in Y_train])),  # 训练集: [(array(784,1), label), ...]
        list(zip(X_dev, Y_dev)),      # 验证集
        list(zip(X_test, Y_test))     # 测试集
    )
    
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

if __name__ == "__main__":
    # 测试数据格式
    (X_train, Y_train), (X_dev, Y_dev), (X_test, Y_test) = load_mnist()
    print("训练集样本形状:", X_train[0].shape)  # 应输出 (784, 1)
    print("标签类型:", type(Y_train[0]))       # 应输出 <class 'numpy.uint8'>
    print("像素值范围:", X_train.min(), X_train.max())  # 应输出 0.0 1.0