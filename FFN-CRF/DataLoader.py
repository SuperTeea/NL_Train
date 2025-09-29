class DataLoader:
    
    def __init__(self, path, type = 'conll', encoding = 'utf-8'):
        self._path = path
        self.encoding = encoding
        self.type = type
        self._sentences = None
        self._words = None
        self._tags = None
        
    
    def _parse(self):
        self._sentences = []
        self._words = set()
        self._tags = set()
        
        with open(self._path, 'r', encoding=self.encoding) as f:
            if self.type != 'conll':    # ctb5
                # raise NotImplementedError('other types have not been implemented.')
                cur = []
                while f.readable():
                    line = f.readline()
                    if len(line) == 4:
                        cur = list()
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
                while f.readable():
                    line = f.readline()
                    if len(line) == 1:
                        if cur: # 非空
                            sentences = [x[0] for x in cur]
                            tags = [x[1] for x in cur]
                            self._sentences.append((sentences, tags))
                        cur = list()
                        continue
                    elif len(line) == 0:
                        break
                    line = line.split('\t')
                    word = line[1]
                    self._words.add(word)
                    tag = line[3]
                    self._tags.add(tag)
                    cur.append((word, tag))
    
    @property
    def sentences(self):
        if self._sentences:
            return self._sentences
        else:
            self._parse()
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
    file_train = r'Data\little-data\dev.conll'
    train_data = DataLoader(file_train)
    print(len(train_data.sentences))
    print(train_data.sentences[0])