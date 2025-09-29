from random import shuffle, seed

seed(1145)    # 设置随机种子

class DataLoader:
    START = ('bos', 'START')
    END = ('eos', 'END')
    
    def __init__(self, path, type = 'conll', encoding = 'utf-8', startpadd = True, endpadd = True, shuff = True):
        self._path = path
        self.encoding = encoding
        self.type = type
        self.startpadd = startpadd
        self.endpadd = endpadd
        self.shuff = shuff
        self._sentences = None
        self._words = None
        self._tags = None
        
    
    def _parse(self):
        self._sentences = []
        self._words = set()
        self._tags = set()
        if self.startpadd:
            self._words.add(DataLoader.START[0])
            self._tags.add(DataLoader.START[1])
        
        with open(self._path, 'r', encoding=self.encoding) as f:
            if self.type != 'conll':    # ctb5
                # raise NotImplementedError('other types have not been implemented.')
                cur = []
                if self.startpadd:
                    cur.append(DataLoader.START)
                while f.readable():
                    line = f.readline()
                    if len(line) == 4:
                        if cur: # 非空
                            if self.endpadd:
                                cur.append(DataLoader.END)
                            sentences = [x[0] for x in cur]
                            tags = [x[1] for x in cur]
                            self._sentences.append((sentences, tags))
                        cur = list()
                        if self.startpadd:
                            cur.append(DataLoader.START)
                        continue
                    elif len(line) == 0:
                        break
                    line = line.split()
                    word = line[1]
                    self._words.add(word)
                    tag = line[3]
                    self._tags.add(tag)
                    cur.append((word, tag))
            else:   # conll
                cur = []
                if self.startpadd:
                    cur.append(DataLoader.START)
                while f.readable():
                    line = f.readline()
                    if len(line) == 1:
                        if cur: # 非空
                            if self.endpadd:
                                cur.append(DataLoader.END)
                            sentences = [x[0] for x in cur]
                            tags = [x[1] for x in cur]
                            self._sentences.append((sentences, tags))
                        cur = list()
                        if self.startpadd:
                            cur.append(DataLoader.START)
                        continue
                    elif len(line) == 0:
                        break
                    line = line.split('\t')
                    word = line[1]
                    self._words.add(word)
                    tag = line[3]
                    self._tags.add(tag)
                    cur.append((word, tag))
                    
    def reshuff(self):
        if self._sentences:
            shuffle(self._sentences)
        else:
            self._parse()
    
    @property
    def sentences(self):
        if self._sentences:
            return self._sentences
        else:
            self._parse()

        if self.shuff:
            shuffle(self.sentences)
        return self._sentences
    
    @property
    def words(self):
        if self._words:
            return self._words
        else:
            self._parse()
        return self._words
    
    @property
    def tags(self):
        if self._tags:
            return self._tags
        else:
            self._parse()
        return self._tags
    
if __name__ == '__main__':
    file_train = r'Data\little-data\train.conll'
    train_data = DataLoader(file_train)
    for batch in train_data.getBatchCorpus():
        print(len(batch))