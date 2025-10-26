from torch.utils.data import Dataset
from functools import reduce

class ConllDataset(Dataset):
    r'''
    适配conll数据的Dataset
    '''
    
    def __init__(self, conll_file):
        self.data = []
        self.word_to_ix = {}
        self.tag_to_ix = {}
        self.ix_to_word = {}
        self.ix_to_tag = {}
        self.__readData(conll_file)
        self.__buildIdxTrans()

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
                    
    def __buildIdxTrans(self):
        sentence, tags = reduce(lambda x,y: (x[0] + y[0], x[1] + y[1]), self.data)
        self.word_to_ix = {word : ix for ix, word in enumerate(set(sentence))}
        self.tag_to_ix = {tag : ix for ix, tag in enumerate(set(tags))}
        self.ix_to_word = {ix : word for word, ix in self.word_to_ix.items()}
        self.ix_to_tag = {ix : tag for tag, ix in self.tag_to_ix.items()}

    def __len__(self):
        return len(self.data)
    
    def revert(self, data):
        '还原idx形式的数据'
        return ([self.ix_to_word[x] for x in data[0]], [self.ix_to_tag[x] for x in data[1]])

    # 这里其实直接存储idx，有需要时候再还原word其实更好
    def __getitem__(self, idx):
        return ([self.word_to_ix[x] for x in self.data[idx][0]], [self.tag_to_ix[x] for x in self.data[idx][1]])
    
if __name__ == '__main__':
    dataset = ConllDataset(r'path')
    print(dataset[0])
    print(dataset.revert(dataset[0]))