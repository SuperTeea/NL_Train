import numpy as np
from Network import Network
from DataLoader import DataLoader
import logging
import time
import random
    
class FFNModel():
    def __init__(self, layers, init_data : DataLoader, logger : logging.Logger = None):
        """
        layer是隐含层的神经元数，第一层和最后一层不用提供，因为这两层决定于训练数据
        e.g. [114,514]
        """
        
        self.logger = logger
        self.feaseq = {}
        self.fearev = {}
        self.tagseq = {}
        self.tagrev = {}
        self.data = init_data
        self.data.startpadd = True
        self.tags = self.data.tags
        self._initFeatureSpace()
        self.model = Network([len(self.feaseq)] + layers + [len(self.tagseq)])
        self.tagnum = len(self.tagseq)
        self.feanum = len(self.feaseq)
        
    
    def _initFeatureSpace(self):
        feature_set = set()
        tag_set = set()
        for sentence, tags in self.data.sentences:
            for tag in tags:
                tag_set.add(tag)
            for i in range(0,len(sentence)):
                for f in self.__class__.templates(sentence, i):
                    feature_set.add(f)
        for ind, f in enumerate(feature_set):
            self.feaseq[f] = ind
            self.fearev[ind] = f
        for ind, tag in enumerate(tag_set):
            self.tagseq[tag] = ind
            self.tagrev[ind] = tag
            
    def output(self, s : str):
        print(s)
        if self.logger:
            self.logger.info(s)
            
    def SGD(self, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            training_data : DataLoader = None,
            evaluation_data : DataLoader =None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        if training_data == None:
            training_data = self.data
            
        training_data = self.process_data(self.data, is_train=True)
        evaluation_data = self.process_data(evaluation_data) if evaluation_data else None
        
        # self.model.SGD(epochs, mini_batch_size, eta, lmbda,train_data,eval_data,
        #                monitor_evaluation_cost,
        #                monitor_evaluation_accuracy, 
        #                monitor_training_cost,
        #                monitor_training_accuracy)

        if evaluation_data: n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for j in range(epochs):
            random.shuffle(training_data)
            st = time.time()
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                mini_batch = self.expand_batch(mini_batch, is_train=True)
                self.model.update_mini_batch(
                    mini_batch, eta, lmbda, len(training_data))
            et = time.time()
            print (f"Epoch {j} training complete, cost {et - st:.1f} second.")
            # if monitor_training_cost:
            #     cost = self.model.total_cost(training_data, lmbda)
            #     training_cost.append(cost)
            #     print ("Cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print ("Accuracy on training data: {} / {}".format(
                    accuracy, n))
            # if monitor_evaluation_cost:
            #     cost = self.model.total_cost(evaluation_data, lmbda, convert=True)
            #     evaluation_cost.append(cost)
            #     print ("Cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy and evaluation_data:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print ("Accuracy on evaluation data: {} / {}".format(
                    accuracy, n_data))
            print
        return evaluation_cost, evaluation_accuracy, \
            training_cost, training_accuracy
        
    def accuracy(self, tot_data, convert=False, mini_batch_size = 30):
        """
        因为不能直接载入内存，所以精确度也要做修改分批计算
        """
        
        ans = 0
        mini_batches = [
        tot_data[k:k+mini_batch_size]
        for k in range(0, len(tot_data), mini_batch_size)]
        for data in mini_batches:
            if convert:
                data = self.expand_batch(data,is_train=True)
                results = [(np.argmax(self.model.feedforward(x)), np.argmax(y))
                        for (x, y) in data]
            else:
                data = self.expand_batch(data)
                results = [(np.argmax(self.model.feedforward(x)), y)
                            for (x, y) in data]
            ans += sum(int(x == y) for (x, y) in results)
        return ans
    
    def process_data(self, data: DataLoader, is_train = False):
        datas = []
        for sentence, tags in data.sentences:
            for i in range(len(sentence)):
                x = [self.feaseq[x] for x in self.templates(sentence, i) if x in self.feaseq]
                y = self.tagseq[tags[i]]
                datas.append((x,y))
        return datas
    
    def expand_batch(self,batch,is_train = False):
        ret = []
        for x,y in batch:
            nx = np.zeros((self.feanum,1))
            for i in x:
                nx[i,0] = 1
            if is_train:
                y = self.vectorized_result(y)
            ret.append((nx,y))
        return ret
                    
    def vectorized_result(self, j):
        """
        对于训练，要求输出是一个向量而非单个结果
        """
        e = np.zeros((self.tagnum, 1))
        e[j] = 1.0
        return e
    
    @classmethod
    def templates(cls, sentence, index):
        length = len(sentence)
        sharp_c = len(sentence[index])
        ret = []
        ret.append('02_' + sentence[index])   # 02 S自身
        if index > 0:
            ret.append('03_' + sentence[index-1])   # 03 前一个词
        else:
            ret.append('03_' + '$$')   # 03 前一个词
            
        if index < length - 1:
            ret.append('04_' + sentence[index+1])   # 04 后一个词
        else:
            ret.append('04_' + '##')
        
        if index > 0:
            ret.append('05_' + sentence[index] + sentence[index-1][-1])   # 05 自身 + 前一个词的最后一个字
        else:
            ret.append('05_' + sentence[index] + '$')   # 05 自身 + 前一个词的最后一个字
        

        if index < length - 1:
            ret.append('06_' + sentence[index] + sentence[index+1][0])   # 06 自身 + 后一个词第一个字
        else:
            ret.append('06_' + sentence[index] + '#')   # 06 自身 + 后一个词第一个字

        ret.append('07_' + sentence[index][0])   # 07 当前词第一个字
        ret.append('08_' + sentence[index][-1])   # 08 当前词最后一个字
        if sharp_c >= 2:   # 空也算特征吧,之后可以调调看(去掉等于)
            ret.append('09_' + sentence[index][1:-1])   # 09 当前词掐头去尾
        if sharp_c >= 1:
            ret.append('10_' + sentence[index][:-1])   # 10 当前词去尾
        if sharp_c >= 1:
            ret.append('11_' + sentence[index][-1] + sentence[index][1:-1])   # 11 当前词尾+中部
        
        if index == 0:
            former_chr = sentence[index - 1][-1]
        else:
            former_chr = '$'
            
        if index == length - 1:
            later_chr = '#'
        else:
            later_chr = sentence[index + 1][0]
        
        if sharp_c == 1:   # 12 当前词+前后一个字符
            ret.append('12_' + sentence[index] + former_chr + later_chr)   

        for i in range(sharp_c - 1):
            if sentence[index][i] == sentence[index][i + 1]:
                ret.append('13_' + sentence[index][i] + 'consecutive')  # 13 连词
        ret.append('14_' + sentence[index][-4:])  # 14 最大四个的后缀
        ret.append('15_' + sentence[index][:4])  # 15 最大四个的前缀
        return ret
    
if __name__ == '__main__':
    file_train = r'Data\little-data\train.conll'
    file_dev = r'Data\little-data\dev.conll'
    large_train = r'Data\ctb5-postagged\train'
    large_dev = r'Data\ctb5-postagged\dev'
    debug_train = DataLoader(r'Data\debug\train.conll')
    train_data = DataLoader(file_train)
    dev_data = DataLoader(file_dev)
    lm = FFNModel([44,33],train_data)
    print(lm.feanum, lm.tagnum)
    # print(list(lm.feaseq.items())[:100])
    lm.SGD(20,30,3,monitor_evaluation_accuracy=True, monitor_training_accuracy=True,evaluation_data=dev_data)
    