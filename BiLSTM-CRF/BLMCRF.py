from torch import nn
import torch
from torchcrf import CRF
import torch.nn.functional as F

class BLMCRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super().__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout()
        self.hidden2tag = nn.LazyLinear(tagset_size)
        self.crf = CRF(tagset_size,batch_first=True)
        
    def forward(self, sentence, gold_tags):
        'crf的特殊性，farward直接计算损失'
        mask = (sentence != 0)
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)
        dropped_tag_space = self.dropout(tag_space)
        ll = -self.crf.forward(dropped_tag_space, gold_tags, mask)  # 负似然才好做损失函数
        return ll
    
    def decode(self, sentence):
        '预测序列对应的tag'
        mask = (sentence != 0)
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out)   # 这个只会用在eval，dropout不需要了
        return self.crf.decode(tag_space, mask)  
        