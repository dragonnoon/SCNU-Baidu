from tqdm import trange
import numpy as np
import torch
import os
from tqdm import tqdm
import jieba
import json
from transformers import BertTokenizer
from model import BertModel

tokenizer = BertTokenizer.from_pretrained('./pretrained_model/chinese-roberta-wwm-ext')
# tokenizer = BertTokenizer.from_pretrained('./pretrained_model/roberta_wwm_ext_pytorch_large')
MAXLEN = 512
# 微调后参数存放位置
SAVE_PATH = './models/roberta.pt'

def load_dataset(dataset_path):
    wenshu_dataset = []
    wenshu_keys = []
    dataset_dir = os.listdir(dataset_path)
    for dir in tqdm(dataset_dir):
        with open(os.path.join(dataset_path, dir), 'r', encoding='utf-8') as f:
            data = json.load(f)
            wenshu_dataset.append(data)
            wenshu_keys.append(dir[:-5])  # 保存key
            f.close()
    return wenshu_dataset, wenshu_keys

def read_stopwords():
    stopwords = []
    with open('./datasets/stopwords.txt', encoding='utf-8') as file:
        for line in file.readlines():
            stopwords.append(line.strip('\n'))

    with open('./datasets/stopword.txt', encoding='utf-8') as file:
        for line in file.readlines():
            stopwords.append(line.strip('\n'))
    stopwords = list(set(stopwords))
    return stopwords

# 判断是否为数字
def not_is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return False
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return False
    except (TypeError, ValueError):
        pass
    return True

def seg_depart(sentence, stopwords):
    # 对文档中的每一行进行中文分词
    sentence_depart = jieba.cut(sentence.strip())
    # 输出结果为outstr
    outstr = ''
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t' and not_is_number(word):
                outstr += word
    return outstr

if __name__ == '__main__':
    device = torch.device("cuda")
    print("\t* Building model...")
    model = BertModel().to(device)
    model.load_state_dict(torch.load(SAVE_PATH))
    # 加载数据集，读取query和候选文本
    dataset_path = './datasets/dev'
    wenshu_dataset_dev, wenshu_dev_keys = load_dataset(dataset_path)
    stopwords = read_stopwords()  # 加载停用词
    wenshu_rank_lists_dev = []
    model.eval()
    with torch.no_grad():
        for i in trange(len(wenshu_dataset_dev)):
            scores = []  # 记录相似度
            data = wenshu_dataset_dev[i]
            query = data['query']  # 查询文本
            # a = query   64.78764
            a = seg_depart(query.replace('\n', '').replace('\r', ''), stopwords)
            a = a.replace("原告", "")
            a = a.replace("被告", "")
            ctxs_key = list(range(100))
            for k in ctxs_key:
                # b = data['ctxs'][str(k)]['JudgeAccusation']   64.78764
                b = seg_depart(data['ctxs'][str(k)]['JudgeAccusation'].replace('\n', '').replace('\r', ''), stopwords)
                names = data['ctxs'][str(k)]['Parties']
                for m in names:
                    if "NameText" in m:
                        b = b.replace(m["NameText"], '')
                    if "Prop" in m:
                        b = b.replace(m["Prop"], '')

                tokens_seq_1 = tokenizer.tokenize(a)
                tokens_seq_2 = tokenizer.tokenize(b)

                if len(tokens_seq_1) > ((MAXLEN - 3) // 2):
                    tokens_seq_1 = tokens_seq_1[0:(MAXLEN - 3) // 2]
                if len(tokens_seq_2) > ((MAXLEN - 3) // 2):
                    tokens_seq_2 = tokens_seq_2[0:(MAXLEN - 3) // 2]

                seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
                seq_segment = [0] * (len(tokens_seq_1) + 2) + [1] * (len(tokens_seq_2) + 1)
                # ID化
                seq = tokenizer.convert_tokens_to_ids(seq)
                # 根据max_seq_len与seq的长度产生填充序列
                padding = [0] * (MAXLEN - len(seq))
                # 创建seq_mask
                seq_mask = [1] * len(seq) + padding
                # 创建seq_segment
                seq_segment = seq_segment + padding
                # 对seq拼接填充序列
                seq += padding
                seqs = torch.tensor([seq], dtype=torch.long).to(device)
                masks = torch.tensor([seq_mask], dtype=torch.long).to(device)
                segments = torch.tensor([seq_segment], dtype=torch.long).to(device)
                labels = torch.tensor([1], dtype=torch.long).to(device)
                loss, logits, probabilities = model(seqs, masks, segments, labels)
                m=probabilities[0][1]
                x=float(probabilities[0][1].cpu().numpy())
                scores.append(x)

            rank_list = np.array(scores).argsort().tolist()[::-1]  # 排序
            wenshu_rank_lists_dev.append(rank_list)

    # 保存结果
    result = {}
    result.update(zip(wenshu_dev_keys, wenshu_rank_lists_dev))

    import json


    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)


    # 保存结果到本地
    result_path = 'result_bm25_dev.json'
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, cls=NpEncoder, indent=4)
        f.close()

