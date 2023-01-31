import torch
import torch.nn as nn

'''
读取数据处理，保存词向量字典
变量:
    vectorPath: 词向量文件路径
    wordVector: 词向量字典
变量结构：
    wordVector: {(word: str) : (vector: tensor embedding dim)}: dict
'''
word2vecPath = r'models/wordVec.csv'
wordVector = {}
with open(word2vecPath, 'r') as w:
    while True:
        varStr = w.readline()
        if varStr == '':
            break
        else:
            varStr = varStr.rstrip('\n')
            varList = varStr.split(',', 1)
            wordVector[varList[0]] = torch.tensor(eval(varList[1]))

'''
参数配置:
    EMBEDDING_DIM: 词向量维度
    HIDDEN_DIM: 隐状态维度
    VOCAB_SIZE: 词向量词典大小
    ATTACH_LENGTH: 单个评论相关数据(包括rating,useful等的其他用户反馈数据)长度
    DEVICE: 选择模型加载于CPU or GPU, DEVICE为 'cuda' 时加载于GPU,为 'cpu' 时加载于CPU
'''
EMBEDDING_DIM = len(list(wordVector.values())[0])
HIDDEN_DIM = 301
VOCAB_SIZE = len(wordVector)
ATTACH_LENGTH = 4
DEVICE = 'cpu'

"""
加载模型的路径配置
"""
LSTM_PATH = 'models/LSTM.pt'
BiLSTM_PATH = 'models/BiLSTM.pt'
GRU_PATH = 'models/GRU.pt'
BiGRU_PATH = 'models/BiGRU.pt'


class GRUTagger(torch.nn.Module):
    """
    定义网络结构
    由于已有现成词向量，本GRU网络结构仅包含一个GRU层，一个线性层，不包含Embedding层
    对于训练集的向量序列(评论向量序列+词向量序列)GRU层的输出，取最后一个状态的输出值作为句子的句向量，通过一个线性层映射到一维空间，
    最后通过Sigmoid()函数的计算，四舍五入后作为模型对是否为虚假评论的判断
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size, bidirectional=False):
        """
        :param embedding_dim: int, 词向量维度
        :param hidden_dim: int, 隐藏层数量,对于长度多于此数量的句子序列,进行截取操作,长度少于此数量的序列不变
                                注： 1. 在模型进行前向传播时,实际传播的不是该长度的数据向量,结果也不是取最后一个状态的输出值作为结果,
                                    而是以句子本身长度为为输入
                                    2. 句子最后一个序列对应的细胞输出作为结果,若为双向,则取两个方向的最后一个隐状态合成为一个向量作为
                                    句向量，后执行线性映射操作
        :param vocab_size: 词典的大小
        :param output_size: 输出维度，该问题中通过线性层将输出映射为一维状态空间作为结果
        """
        super(GRUTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.GRU = nn.GRU(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=self.bidirectional)
        if self.bidirectional is False:
            self.out2tag = nn.Linear(self.hidden_dim, self.output_size)
        else:
            self.out2tag = nn.Linear(2 * self.hidden_dim, self.output_size)
        self.hidden = self.init_hidden()
        self.forwardVector = None

    def init_hidden(self):
        """
        该方法用于输出清空隐状态的值.由于模型在传播后不会自动将状态置零,该函数返回一个供模型隐状态置零的张量.
        :return : (tensor: zeros, tensor: zeros): tuple
        """
        return (torch.zeros(1, 1, self.hidden_dim).to(DEVICE),
                torch.zeros(1, 1, self.hidden_dim).to(DEVICE))

    def forward(self, tensors):
        """
        :param :packedSequence[BATCH*HIDDEN_DIM*EMBEDDING_DIM]
        :return: tags: tensor[BATCH*1*1], 对批次中每个数据的传播结果
        """
        # 前向传播
        out, self.hidden = self.GRU(tensors)
        if self.bidirectional is False:
            # 将输出数据解包
            out, index = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # 取句子序列的最后一个状态
            tag_space = out[[i for i in range(len(index))], [i - 1 for i in index]]
            # 把状态值作为该批次所有句子对应的向量
            self.forwardVector = tag_space
        else:
            # 将输出数据解包
            out, index = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # 取正向输出的最后一个状态,反向输出的第一个状态
            tag_Forward = out[[i for i in range(len(index))], [i - 1 for i in index], :HIDDEN_DIM]
            tag_Backward = out[[i for i in range(len(index))], [0] * len(index), HIDDEN_DIM:]
            # 把两个状态合并为一个向量,准备作为线性层输入
            tag_space = torch.cat([tag_Forward, tag_Backward], dim=1)
            self.forwardVector = tag_space
        # 通过线性层将数据转为1维
        tags = self.out2tag(tag_space)
        return tags


class LSTMTagger(torch.nn.Module):
    """
    定义网络结构
    由于已有现成词向量，本LSTM网络结构仅包含一个LSTM层，一个线性层，不包含Embedding层
    对于训练集的向量序列(评论向量序列+词向量序列)LSTM层的输出，取最后一个状态的输出值作为句子的句向量，通过一个线性层映射到一维空间，
    最后通过Sigmoid()函数的计算，四舍五入后作为模型对是否为虚假评论的判断
    """

    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size, bidirectional=False):
        """
        :param embedding_dim: int, 词向量维度
        :param hidden_dim: int, 隐藏层数量,对于长度多于此数量的句子序列,进行截取操作,长度少于此数量的序列不变
                                注： 1. 在模型进行前向传播时,实际传播的不是该长度的数据向量,结果也不是取最后一个状态的输出值作为结果,
                                    而是以句子本身长度为为输入
                                    2. 句子最后一个序列对应的细胞输出作为结果,若为双向,则取两个方向的最后一个隐状态合成为一个向量作为
                                    句向量，后执行线性映射操作
        :param vocab_size: 词典的大小
        :param output_size: 输出维度，该问题中通过线性层将输出映射为一维状态空间作为结果
        """
        super(LSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.LSTM = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=self.bidirectional)
        if self.bidirectional is False:
            self.out2tag = nn.Linear(self.hidden_dim, self.output_size)
        else:
            self.out2tag = nn.Linear(2 * self.hidden_dim, self.output_size)
        self.hidden = self.init_hidden()
        self.forwardVector = None

    def init_hidden(self):
        """
        该方法用于输出清空隐状态的值.由于模型在传播后不会自动将状态置零,该函数返回一个供模型隐状态置零的张量.
        :return : (tensor: zeros, tensor: zeros): tuple
        """
        return (torch.zeros(1, 1, self.hidden_dim).to(DEVICE),
                torch.zeros(1, 1, self.hidden_dim).to(DEVICE))

    def forward(self, tensors):
        """
        :param :packedSequence[BATCH*HIDDEN_DIM*EMBEDDING_DIM]
        :return: tags: tensor[BATCH*1*1], 对批次中每个数据的传播结果
        """
        # 前向传播
        out, self.hidden = self.LSTM(tensors)
        if self.bidirectional is False:
            # 将输出数据解包
            out, index = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # 取句子序列的最后一个状态
            tag_space = out[[i for i in range(len(index))], [i - 1 for i in index]]
            # 把状态值作为该批次所有句子对应的向量
            self.forwardVector = tag_space
        else:
            # 将输出数据解包
            out, index = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
            # 取正向输出的最后一个状态,反向输出的第一个状态
            tag_Forward = out[[i for i in range(len(index))], [i - 1 for i in index], :HIDDEN_DIM]
            tag_Backward = out[[i for i in range(len(index))], [0] * len(index), HIDDEN_DIM:]
            # 把两个状态合并为一个向量,准备作为线性层输入
            tag_space = torch.cat([tag_Forward, tag_Backward], dim=1)
            self.forwardVector = tag_space
        # 通过线性层将数据转为1维
        tags = self.out2tag(tag_space)
        return tags
