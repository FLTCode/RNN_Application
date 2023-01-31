from Configs import *

GRUModel = torch.load(GRU_PATH, map_location=torch.device(DEVICE))
BiGRUModel = torch.load(BiGRU_PATH, map_location=torch.device(DEVICE))
LSTMModel = torch.load(LSTM_PATH, map_location=torch.device(DEVICE))
BiLSTMModel = torch.load(BiLSTM_PATH, map_location=torch.device(DEVICE))


def data2tensor(data):
    """
    以下函数把传入的单条数据转为压紧后的数据向量(将对评论数据进行补零至EMBEDDING_DIM处理)
    :param data: str, 内容包含(无标点的句子,句子相关数据)
    :returns: tensors: packedSequence, 对(评论数据 + 句子)转化为向量，之后补零并压紧得到的"变长向量"
                   注:数据的第一列为经过补零后的评论数据张量
    """
    varStr = data.rstrip('\n')
    varList = []
    List = varStr.split(",")
    varList.append(List[0].split(' '))
    varList.append([])
    for i in List[1:]:
        varList[1].append(eval(i))
    tensors = []
    lengths = [len(varList[0])]
    varList = [varList]
    attachFiller = [0] * (EMBEDDING_DIM - ATTACH_LENGTH)  # 用于给评论相关数据补零.
    adjust = [*(varList[0][1]), *attachFiller]
    zeroVec = [0]*EMBEDDING_DIM
    seqTensor = [(wordVector[word].tolist() if word in wordVector else zeroVec) for word in varList[0][0]]
    if len(varList[0][0]) < HIDDEN_DIM - 1:
        filler = [[0] * EMBEDDING_DIM] * (HIDDEN_DIM - len(varList[0][0]) - 1)
        tensors.append([adjust, *seqTensor, *filler])
    else:
        lengths[0] = 300
        tensors.append([adjust, *seqTensor[:HIDDEN_DIM - 1]])
    return nn.utils.rnn.pack_padded_sequence(torch.tensor(tensors, dtype=torch.float32),
                                             torch.tensor(lengths, dtype=torch.int64),
                                             batch_first=True,
                                             enforce_sorted=False).to(DEVICE)


def evaluation(service: str, data: str):
    """
    :param service: str, 选用何种模型预测,可选项为 'LSTM', 'BiLSTM', 'GRU', 'BiGRU'
    :param data: str, 要送入的数据, 送入数据的结构请参照用例
    :return: bool, 标识对该数据的判断是真还是假, True为真, False为假
    用例:
    input:
        varStr = 'Children less than 10 banned for brunch parents beware,2,0,0,0'
        print(evaluation('LSTM', varStr))
    output:
        True
    注意：如果函数运行保错，修改modelConfigs.py中的device变量, 切换模型的加载位置至cpu
    """
    data_in = data2tensor(data)
    if service == 'LSTM':
        return LSTMModel(data_in).tolist()[0][0] > 0
    elif service == 'BiLSTM':
        return BiLSTMModel(data_in).tolist()[0][0] > 0
    elif service == 'GRU':
        return GRUModel(data_in).tolist()[0][0] > 0
    else:
        return BiGRUModel(data_in).tolist()[0][0] > 0
