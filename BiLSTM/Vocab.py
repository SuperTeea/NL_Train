from functools import reduce
from ConllDataset import ConllDataset

class Vocab():
    '''
    一个代表词汇的类，可以转换ConllDataset输出的word为idx
    '''
    def __init__(self, dataset : ConllDataset):
        self.word_to_ix = {}
        self.tag_to_ix = {}
        self.ix_to_word = {}
        self.ix_to_tag = {}
        self.__buildIdxTrans(dataset.data)
    
    def __buildIdxTrans(self, dataset):
        sentence, tags = reduce(lambda x,y: (x[0] + y[0], x[1] + y[1]), dataset)
        self.word_to_ix = {word : ix + 2 for ix, word in enumerate(set(sentence))}
        self.tag_to_ix = {tag : ix + 2 for ix, tag in enumerate(set(tags))}
        self.word_to_ix['<PAD>'] = 0
        self.word_to_ix['<UNK>'] = 1
        self.tag_to_ix['<PAD>'] = 0
        self.tag_to_ix['<UNK>'] = 1
        self.ix_to_word = {ix : word for word, ix in self.word_to_ix.items()}
        self.ix_to_tag = {ix : tag for tag, ix in self.tag_to_ix.items()}
        
    def revert(self, data):
        '还原idx形式的数据'
        return ([self.ix_to_word[x] for x in data[0]], [self.ix_to_tag[x] for x in data[1]])
    
    def encode(self, data):
        '把word,tag编码成idx'
        return ([self.word_to_ix.get(x,1) for x in data[0]], [self.tag_to_ix.get(x,1) for x in data[1]])
    
    def encodeSentence(self, sentence):
        '编码 word 序列'
        return [self.word_to_ix.get(x,1) for x in sentence]
    
    def decodeSentence(self, sentence):
        '解码 word idx 序列'
        return [self.ix_to_word[x] for x in sentence]
    
    def encodeTags(self, tags):
        '编码 word 序列'
        return [self.tag_to_ix.get(x,1) for x in tags]
    
    def decodeSentence(self, tags):
        '解码 word idx 序列'
        return [self.ix_to_tag[x] for x in tags]
    
    def wordSize(self):
        return len(self.word_to_ix)
    
    def tagSize(self):
        return len(self.tag_to_ix)

if __name__ == '__main__':
    dataset = ConllDataset(r'path')
    vocab = Vocab(dataset)
    print(list(vocab.word_to_ix.items())[:10])