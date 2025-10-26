from Trainer import Trainer
from ConllDataset import ConllDataset
from BiLSTMnet import BiLSTM
from Vocab import Vocab
from torch.optim import SGD

if __name__ == '__main__':
    train_dataset = ConllDataset(r'..\Data\little-data\train.conll')
    test_dataset = ConllDataset(r'..\Data\little-data\dev.conll')
    # train_dataset = ConllDataset(r'..\Data\little-data\dev.conll')
    # test_dataset = ConllDataset(r'..\Data\little-data\train.conll')
    vocab = Vocab(train_dataset)
    train_dataset.confTrans(vocab.encode)   # u1s1我的这种设计模式耦合多，还是蛮糟糕的，先凑合
    test_dataset.confTrans(vocab.encode)   
    model = BiLSTM(128, 128, vocab.wordSize(), vocab.tagSize())
    optimizer = SGD(model.parameters(), lr = 0.5)
    trainer = Trainer(model, optimizer)
    trainer.fit(train_dataset,batchsize=8, epoches=64, test_dataset=test_dataset)