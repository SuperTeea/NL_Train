import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

class Trainer:
    def __init__(self, model, optimizer:torch.optim.Optimizer, device = 'cuda', loss = torch.nn.NLLLoss(ignore_index=0)):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss = loss
        
    def train_step(self, batch):
        X , y = batch
        X:torch.Tensor
        X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        y_out = self.model(X)   # 这里是 X 而非 batch
        
        l = self.loss(y_out.reshape(-1,y_out.size()[-1]), y.reshape(-1))
        l:torch.Tensor
        l.backward()
        
        self.optimizer.step()
        return l.item()
    
    def train_epoch(self, dataloader : DataLoader):
        self.model.train()
        tot_loss = 0
        for batch in dataloader:
            tot_loss += self.train_step(batch)
        return tot_loss / len(dataloader)
    
    def collate_fn(self, batch):
        sequences, labels = zip(*batch)
        return pad_sequence(sequences, batch_first=True), pad_sequence(labels, batch_first=True)
    
    def fit(self, dataset, epoches = 10, batchsize = 32, test_dataset = None):
        dataloader = DataLoader(dataset,batchsize,shuffle=True,collate_fn=self.collate_fn)
        for epoch in range(epoches):
            l = self.train_epoch(dataloader)
            if test_dataset:
                acc, tot = self.evaluate(test_dataset)
            print(f'epoch {epoch} : loss {l:.4f} {acc} / {tot} acc {acc / tot:.4f}' if test_dataset else f'epoch {epoch} : loss {l:.4f}')
            
    def evalutaeBatch(self, batch):
        '评估一个batch,返回准确的个数'
        X , y = batch
        mask = (y != 0)
        X:torch.Tensor
        X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        y_out = self.model(X)   # 这里是 X 而非 batch
        return ((torch.argmax(y_out, dim=-1) == y) & mask).sum().item(), mask.sum().item()  # 这里注意运算优先级
        
    def evaluate(self, dataset, batchsize = 32):
        self.model.eval()
        dataloader = DataLoader(dataset,batchsize,shuffle=True,collate_fn=self.collate_fn)
        tot_acc = 0
        tot_cnt = 0
        with torch.no_grad():
            for batch in dataloader:
                acc, valid = self.evalutaeBatch(batch)
                tot_acc += acc
                tot_cnt += valid
        return tot_acc , tot_cnt