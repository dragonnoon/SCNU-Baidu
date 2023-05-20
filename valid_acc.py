import os
import torch
from torch.utils.data import DataLoader
from data import DataPrecessForSentence
from utils import train, validate
from transformers import BertTokenizer
from model import BertModel
from transformers.optimization import AdamW
from torch.utils.data import DataLoader, Dataset
from typing import List
import random
from hanziconv import HanziConv
import jsonlines

tokenizer = BertTokenizer.from_pretrained('./pretrained_model/roberta_wwm_ext_pytorch_large')
MAXLEN = 512
SAVE_PATH = './models/roberta_large.pt'
LQCMC_TRAIN = './datasets/train_process1.txt'
LQCMC_DEV = './datasets/test_process.txt'


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


def main(target_dir,
         epochs=10,
         BATCH_SIZE=32,
         lr=2e-05,
         patience=3,
         max_grad_norm=10.0,
         checkpoint=None):
    # checkpoint='./models/best.pth.tar'

    device = torch.device("cuda")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = load_data(LQCMC_TRAIN)
    train_dataloader = DataLoader(LCQMC_Dataset(train_data), shuffle=True, batch_size=BATCH_SIZE)
    print("\t* Loading validation data...")
    dev_data = load_data(LQCMC_DEV)
    dev_dataloader = DataLoader(LCQMC_Dataset(dev_data), shuffle=True, batch_size=BATCH_SIZE)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    model = BertModel().to(device)
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
    # optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.85, patience=0)
    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        print("checkpoint =", checkpoint)
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
        print("epochs_count =", epochs_count)
        print("train_losses =", train_losses)
        print("valid_losses =", valid_losses)
    # Compute loss and accuracy before starting (or resuming) training.
    model.load_state_dict(torch.load(SAVE_PATH))
    _, valid_loss, valid_accuracy, auc = validate(model, dev_dataloader)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}".format(valid_loss,
                                                                                               (valid_accuracy * 100),
                                                                                               auc))

if __name__ == "__main__":
    main("models")
