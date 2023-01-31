import signal
import matplotlib.pyplot as plt
import torch.optim as optim
from Configs import *

'''
读取数据处理，保存数据集到data
变量:
    dataPath: 数据集文件路径
    vectorPath: 词向量文件路径
    data: 数据集
    wordVector: 词向量字典
一些变量结构：
    data: [words of sentence: [word: str]: list, 
           rating, usefulCount, coolCount,funnyCount: int 
           label: bool]: list
    wordVector: {(word: str) : (vector: tensor x dim)}: dict
'''

dataPath = r'TrainSet.csv'
data = []
with open(dataPath, 'r') as d:
    while True:
        varStr = d.readline()
        if varStr == '':
            break
        else:
            varStr = varStr.rstrip('\n')
            varList = []
            List = varStr.split(",")
            varList.append(List[0].split(' '))
            for i in List[1:-1]:
                varList.append(eval(i))
            varList.append(1 if List[-1] == 'Y' else 0)
            data.append(varList)

'''
参数初始化:
    MODEL_TYPE: 模型类型,可选项为LSTM和GRU
    Bidirectional: 是否选择双向神经网络
    EMBEDDING_DIM: 词向量维度
    HIDDEN_DIM: 隐状态维度
    VOCAB_SIZE: 词向量词典大小
    DATA_SIZE: 训练集大小
    SAVE_PATH: 模型训练完成后保存的位置
    LEARNING_RATE: 学习率
    TEST_RAE: 当训练集正确率达到多少时终止训练
    TEST_RATE_INF: 边迭代边计算训练集的正确率，当正确率高于此值对整个训练集进行测试
'''
MODEL_TYPE = 'LSTM'
Bidirectional = False
EMBEDDING_DIM = len(list(wordVector.values())[0])
HIDDEN_DIM = 301
VOCAB_SIZE = len(wordVector)
DATA_SIZE = len(data)
SAVE_PATH = 'model_trains/' + ('Bi' if Bidirectional else '') + MODEL_TYPE + '.pt'
BATCH_SIZE = 100
LEARNING_RATE = 0.001
TEST_RATE = 0.999
TEST_RATE_INF = int(TEST_RATE * 10) / 10


def findHiddenDim(threshold):
    """
    以下函数找到合适的隐藏层数量HIDDEN_DIM
    :param :threshold: 覆盖数据集比例(involvedRate)的阈值
    :return : (HIDDEN_DIM: int, involvedRate: float) :tuple
    在该数据集中，隐藏层数为267时，覆盖数据量已达90%以上，隐藏层数为367时，覆盖数据量已达95%以上
    故本模型集采用301层隐藏层
    """
    varDict = {}
    for k in data:
        varInt = len(k[0])
        varDict[varInt] = varDict.get(varInt, 0) + 1
    varInt = 0
    for k in range(DATA_SIZE):
        varInt = varInt + varDict.get(k, 0)
        if varInt / DATA_SIZE >= threshold:
            return k, varInt / DATA_SIZE


def data2tensor(sequenceList: list, attach: list):
    """
    以下函数把传入的分词序列和评论数据转为压缩后的向量(将对评论数据进行补零至EMBEDDING_DIM处理)
    :param sequenceList: [word: str]: list, 分词序列
    :param attach: [data: int]: list, 评论数据
    :returns : tensors: packedSequence, 对(评论数据 + 句子)转化为向量，之后补零并压紧得到的"变长向量"
               注：数据的第一列为经过补零后的评论数据张量
    """
    tensors = []
    lengths = []
    for index, sequence in enumerate(sequenceList):
        adjust = [*(attach[index]), *attachFiller]
        seqTensor = [wordVector[word].tolist() for word in sequence]
        if len(sequence) < HIDDEN_DIM - 1:
            lengths.append(len(sequence) + 1)
            filler = [[0] * EMBEDDING_DIM] * (HIDDEN_DIM - len(sequence) - 1)
            tensors.append([adjust, *seqTensor, *filler])
        else:
            lengths.append(HIDDEN_DIM)
            tensors.append([adjust, *seqTensor[:HIDDEN_DIM - 1]])
    return nn.utils.rnn.pack_padded_sequence(torch.tensor(tensors, dtype=torch.float32),
                                             torch.tensor(lengths, dtype=torch.int64),
                                             batch_first=True,
                                             enforce_sorted=False).to('cuda')


# 设置模型保存路径,实例化模型对象,声明损失函数(带有sigmoid函数的交叉熵函数)和优化策略(Adam优化器),
# 并将它们送入GPU准备训练
model = eval(MODEL_TYPE + 'Tagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE, 1, bidirectional=Bidirectional)')
loss_function = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
model.to('cuda')
loss_function.to('cuda')


def Test():
    """
    该函数实现对整个训练数据集求准确率
    屏幕将输出: correctRate: float, 整个训练数据集的准确率
    """
    correct = 0
    for k in range(BATCH_GROUPS):
        batch = data[k * BATCH_SIZE:((k + 1) * BATCH_SIZE) if ((k + 1) * BATCH_SIZE) < DATA_SIZE else DATA_SIZE]
        model.zero_grad()
        model.hidden = model.init_hidden()
        commentData = []
        attachData = []
        dataTag = []
        for b in batch:
            commentData.append(b[0])
            attachData.append(b[1:-1])
            dataTag.append([b[-1]])
        tensor = data2tensor(commentData, attachData)
        tag_Forward = model.forward(tensor)
        tag_Forward = tag_Forward >= 0
        tag_Data = torch.tensor(dataTag, dtype=torch.bool).to('cuda')
        correct += len(tag_Data[tag_Forward == tag_Data])
    return correct / DATA_SIZE


def IR_TO_SAVE(sigint, frame):
    """
    实现用户中断以保存模型，当用户通过运行终端中断时, 模型自动保存，并测试数据集准确率
    :return:
    """
    print(f"用户手动中断，开始保存模型...")
    if model.bidirectional is False:
        torch.save(model, SAVE_PATH)
    else:
        torch.save(model, SAVE_PATH)
    print("保存完成，开始测试训练集准确率")
    print(f'所保存的模型在训练集上的准确率为{Rate}')
    print("展示Loss变化趋势图...")
    lossView(Loss)
    exit(0)


# 预置中断信号, 初始化某些数据, 准备开始训练
signal.signal(signal.SIGINT, IR_TO_SAVE)
print('开始训练...')
BATCH_GROUPS = DATA_SIZE // BATCH_SIZE + int(DATA_SIZE % BATCH_SIZE > 0)
attachFiller = [0] * (EMBEDDING_DIM - len(data[0][1:-1]))

torch.manual_seed(2024)
epoch = 0
Loss = []
# 开始训练
while True:
    epoch += 1
    print(f"进行第{epoch}次迭代")
    dataTag = []
    tag_Forward = []
    Loss_Batch = []
    correct = 0
    for k in range(BATCH_GROUPS):
        batch = data[k * BATCH_SIZE:((k + 1) * BATCH_SIZE) if ((k + 1) * BATCH_SIZE) < DATA_SIZE else DATA_SIZE]
        model.zero_grad()
        model.hidden = model.init_hidden()
        commentData = []
        attachData = []
        dataTag = []
        for b in batch:
            commentData.append(b[0])
            attachData.append(b[1:-1])
            dataTag.append([b[-1]])
        tensor = data2tensor(commentData, attachData)
        tag_Forward = model.forward(tensor)
        tag_Data = torch.tensor(dataTag, dtype=torch.float32)
        loss = loss_function(tag_Forward, tag_Data.to('cuda'))
        loss.backward()
        optimizer.step()
        tag_Forward = tag_Forward >= 0
        tag_Data = torch.tensor(dataTag, dtype=torch.bool).to('cuda')
        correct += len(tag_Data[tag_Forward == tag_Data])
        print(f'Loss = {loss.item()}')
        Loss_Batch.append(float(loss.item()))
    Loss_Batch.sort(reverse=True)
    Loss.append(Loss_Batch[int(len(Loss_Batch) / 2)])
    Rate = correct / DATA_SIZE
    print(f'当前历史合格率Rate={Rate}')
    if Rate > TEST_RATE_INF:
        print('历史合格率达到标准，开始测试整个训练集')
        Rate = Test()
        if Rate > TEST_RATE:
            print(f'训练集合格率达到标准，合格率为{Rate}')
            break
        else:
            print(f'训练集合格率未达到标准，合格率为{Rate}')


def lossView(Loss: list):
    plt.plot(Loss, 'oc-')
    plt.title('Process data: Loss-Steps')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.show()


print("训练完成，开始保存")
torch.save(model, SAVE_PATH)
print("保存完成")
print('展示Loss变化趋势图...')
lossView(Loss)
"""
经过43次迭代后GRU模型达到期望效果，训练集正确率99.915%
经过17次迭代后BiGRU模型达到期望效果，训练集正确率为99.923%
"""
