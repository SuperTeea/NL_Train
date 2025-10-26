import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class Trainer:
    def __init__(self, model, optimizer:torch.optim.Optimizer, device = 'cuda', loss = F.cross_entropy):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss = loss
        
    def train_step(self, batch):
        X , y = batch
        X:torch.Tensor
        X.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        y_out = self.model(batch)
        
        l = self.loss(y_out, y)
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
    
    def fit(self, dataset, epoches = 10, batchsize = 32):
        dataloader = DataLoader(dataset,batchsize,shuffle=True)
        for epoch in epoches:
            l = self.train_epoch(dataloader)
            print(f'epoch {epoch} : loss {l}')