from Trainer import Trainer
from ConllDataset import ConllDataset
from BLMCRF import BLMCRF
from Vocab import Vocab
from torch.optim import SGD

if __name__ == '__main__':
    # train_dataset = ConllDataset(r'..\Data\ctb5-postagged\train')
    # test_dataset = ConllDataset(r'..\Data\ctb5-postagged\test')
    train_dataset = ConllDataset(r'..\Data\little-data\train.conll')
    test_dataset = ConllDataset(r'..\Data\little-data\dev.conll')
    vocab = Vocab(train_dataset)
    train_dataset.confTrans(vocab.encode)
    test_dataset.confTrans(vocab.encode)   
    model = BLMCRF(256, 128, vocab.wordSize(), vocab.tagSize())
    optimizer = SGD(model.parameters(), lr = 0.6)
    trainer = Trainer(model, optimizer)
    trainer.fit(train_dataset,batchsize=32, epoches=64, test_dataset=test_dataset)