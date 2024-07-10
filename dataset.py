import torch
from torch.utils.data import Dataset

# 德语转英语数据集
class TranslateDataset(Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(TranslateDataset, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return len(self.enc_inputs)

    def __getitem__(self, idx):
        # an item like [1,2,3,4,0,0]
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

def make_data(sentences, src_vocab, tgt_vocab):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [src_vocab[n] for n in sentences[i][0].split()]
        dec_input = [tgt_vocab[n] for n in sentences[i][1].split()]
        dec_output = [tgt_vocab[n] for n in sentences[i][2].split()]

        enc_inputs.append(enc_input)
        dec_inputs.append(dec_input)
        dec_outputs.append(dec_output)

    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)
