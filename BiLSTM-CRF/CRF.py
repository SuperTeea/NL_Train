from torch import nn
import torch

class CRF(nn.Module):
    '自建CRF模块,权重初始化用默认正态,bias全0'
    '目前还没实现,先准备用库CRF搭框架'
    def __init__(self, tag_num):
        super().__init__()
        self.trans = nn.Parameter(torch.randn(tag_num, tag_num))
    
    def _cal_forward(self):
        pass
    
    def predict(self):
        pass
    
    def forward(self, input):
        pass