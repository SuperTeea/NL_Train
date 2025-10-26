from torch.utils.data import Dataset

class ConllDataset(Dataset):
    r'''
    适配conll数据的Dataset
    '''
    
    def __init__(self, conll_file):
        self.data = []
        self.__readData(conll_file)

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
                

    def __len__(self):
        return len(self.data)
    


    # 这里其实直接存储idx，有需要时候再还原word其实更好
    def __getitem__(self, idx):
        return ([self.word_to_ix[x] for x in self.data[idx][0]], [self.tag_to_ix[x] for x in self.data[idx][1]])
    
if __name__ == '__main__':
    dataset = ConllDataset(r'path')
    print(dataset[0])
    print(dataset.revert(dataset[0]))