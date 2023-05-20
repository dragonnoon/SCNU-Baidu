import os
import torch
import time
import torch.nn as nn
from tqdm import tqdm
from utils import validate, correct_predictions
from transformers import BertTokenizer
from model import BertModel
from transformers.optimization import AdamW
from torch.utils.data import DataLoader, Dataset
from typing import List
from hanziconv import HanziConv
from torch.optim.swa_utils import AveragedModel, SWALR

tokenizer = BertTokenizer.from_pretrained('./pretrained_model/chinese-roberta-wwm-ext')
MAXLEN = 512
PATH = './models/roberta1.pt'
SAVE_PATH = './models/roberta_SWA.pt'
LQCMC_TRAIN = './datasets/train_process1.txt'
LQCMC_DEV = './datasets/test_process1.txt'


def load_data(path: str) -> List:
    sen1 = []
    sen2 = []
    labels = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            sen1.append(line.strip().split('\t')[0])
            sen2.append(line.strip().split('\t')[1])
            labels.append(int(line.strip().split('\t')[2]))
    data = []
    data.append(sen1)
    data.append(sen2)
    data.append(labels)
    return data


class LCQMC_Dataset(Dataset):
    """训练数据集, 重写__getitem__和__len__方法
    """

    def __init__(self, data: List):
        self.max_seq_len = MAXLEN
        self.seqs, self.seq_masks, self.seq_segments, self.labels = self.text_2_id(data)

    def __len__(self):
        return len(self.labels)

    def text_2_id(self, data: str):
        sentences_1 = map(HanziConv.toSimplified, data[0])
        sentences_2 = map(HanziConv.toSimplified, data[1])
        labels = data[2]
        tokens_seq_1 = list(map(tokenizer.tokenize, sentences_1))
        tokens_seq_2 = list(map(tokenizer.tokenize, sentences_2))
        result = list(map(self.trunate_and_pad, tokens_seq_1, tokens_seq_2))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.Tensor(seqs).type(torch.long), torch.Tensor(seq_masks).type(torch.long), torch.Tensor(
            seq_segments).type(torch.long), torch.Tensor(labels).type(torch.long)

    def trunate_and_pad(self, tokens_seq_1, tokens_seq_2):
        # 对超长序列进行截断
        if len(tokens_seq_1) > ((self.max_seq_len - 3) // 2):
            tokens_seq_1 = tokens_seq_1[0:(self.max_seq_len - 3) // 2]
        if len(tokens_seq_2) > ((self.max_seq_len - 3) // 2):
            tokens_seq_2 = tokens_seq_2[0:(self.max_seq_len - 3) // 2]
        # 分别在首尾拼接特殊符号
        seq = ['[CLS]'] + tokens_seq_1 + ['[SEP]'] + tokens_seq_2 + ['[SEP]']
        seq_segment = [0] * (len(tokens_seq_1) + 2) + [1] * (len(tokens_seq_2) + 1)
        # ID化
        seq = tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = seq_segment + padding
        # 对seq拼接填充序列
        seq += padding
        return seq, seq_mask, seq_segment

    def __getitem__(self, idx: int):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[idx], self.labels[idx]


def main(epochs=12,
         BATCH_SIZE=72,
         lr=1e-05,
         max_grad_norm=10.0):
    device = torch.device("cuda")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = load_data(LQCMC_TRAIN)
    train_dataloader = DataLoader(LCQMC_Dataset(train_data), shuffle=True, batch_size=BATCH_SIZE)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = BertModel().to(device)
    model.load_state_dict(torch.load(PATH))
    # -------------------- Preparation for training  ------------------- #
    # 待优化的参数
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, anneal_strategy="linear", anneal_epochs=7, swa_lr=lr*0.4)
    start_epoch = 1
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training roberta model on device: {}".format(device), 20 * "=")
    for epoch in range(start_epoch, epochs + 1):
        print("* Training epoch {}:".format(epoch))

        # Switch the model to train mode.
        model.train()
        device = model.device
        tqdm_batch_iterator = tqdm(train_dataloader)
        for batch_index, (batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels) in enumerate(
                tqdm_batch_iterator):
            # Move input and output data to the GPU if it is used.
            seqs, masks, segments, labels = batch_seqs.to(device), batch_seq_masks.to(device), batch_seq_segments.to(
                device), batch_labels.to(device)
            optimizer.zero_grad()
            loss, logits, probabilities = model(seqs, masks, segments, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        # Update the SWA optimizer's learning rate with the scheduler.
        swa_model.update_parameters(model)
        swa_scheduler.step()
        print()

    torch.optim.swa_utils.update_bn(train_dataloader, swa_model)
    torch.save(swa_model.state_dict(), SAVE_PATH)
    print("SWA finished!")

if __name__ == "__main__":
    main()
