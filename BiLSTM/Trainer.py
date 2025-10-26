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
    
    def fit(self, dataset, epoches = 10, batchsize = 32):
        dataloader = DataLoader(dataset,batchsize,shuffle=True,collate_fn=self.collate_fn)
        for epoch in range(epoches):
            l = self.train_epoch(dataloader)
            print(f'epoch {epoch} : loss {l}')