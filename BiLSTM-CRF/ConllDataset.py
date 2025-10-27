from torch.utils.data import Dataset
from functools import partial
import torch

class ConllDataset(Dataset):
    r'''
    适配conll数据的Dataset
    '''
    
    def __init__(self, conll_file):
        self.data = []
        self.__readData(conll_file)
        self.trans = lambda x: x

    def __readData(self, conll_file):
        sentence = []
        tags = []
        with open(conll_file, encoding='utf-8') as f:
            for row in f.readlines():
                if not row.strip() and sentence:
                    self.data.append((sentence, tags))
                    sentence, tags = list(), list()
                else:
                    row = row.split()
                    sentence.append(row[1])
                    tags.append(row[3])
                
    def confTrans(self, trans):
        self.trans = trans
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return map(torch.tensor, self.trans(self.data[idx]))
    
if __name__ == '__main__':
    dataset = ConllDataset(r'path')
    print(dataset[0])
    print(dataset.revert(dataset[0]))