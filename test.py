import torch
import torch.utils.data as data
import dataset
from model import Transformer
import matplotlib.pyplot as plt

# -----------------dataset-----------------------
# S <START> E <END> P <PAD>

sentences = [
    #    encoder_input          decoder_input        decoder_output
    ['我 想 要 一 杯 啤 酒 。', 'S i want a beer .', 'i want a beer . E'],
    ['我 想 要 一 杯 可 乐 。', 'S i want a coke .', 'i want a coke . E']
]

# source vocabarary
src_vocab = {'P': 0, '我': 1, '想': 2, '要': 3, '一': 4, '杯': 5, '啤': 6, '酒': 7, '可': 8, '乐': 9, '。': 10}
src_vocab_size = len(src_vocab)
src_idx2word = {i: w for w, i in src_vocab.items()}

# target vocabulary
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
tgt_vocab_size = len(tgt_vocab)
tgt_idx2word = {i: w for w, i in tgt_vocab.items()}

src_len = 8  # length of encoder input
tgt_len = 6  # length of decoder input, output

enc_inputs, dec_inputs, dec_outputs = dataset.make_data(sentences, src_vocab, tgt_vocab)
translate_dataset = dataset.TranslateDataset(enc_inputs, dec_inputs, dec_outputs)
loader = data.DataLoader(dataset=translate_dataset, batch_size=2)

# -------------------model----------------------
# Transformer Parameters
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

model = Transformer(src_vocab_size, tgt_vocab_size, n_layers, d_model, n_heads, d_ff)
model = model.to('cuda')
model.load_state_dict(torch.load('transformer.pth'))

# -------------------plot attention----------------------
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_all_attention_weights(attns, src_tokens, tgt_tokens, file_name):
    n_layers = len(attns)
    n_heads = attns[0].size(1)

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(n_heads * 4, n_layers * 4), constrained_layout=True)

    for layer in range(n_layers):
        for head in range(n_heads):
            ax = axes[layer, head]
            cax = ax.matshow(attns[layer][0, head].cpu().detach().numpy(), cmap='viridis')
            ax.set_xticks(range(len(src_tokens)))
            ax.set_yticks(range(len(tgt_tokens)))
            ax.set_xticklabels(src_tokens, rotation=90, fontsize=20)
            ax.set_yticklabels(tgt_tokens, fontsize=20)
            ax.set_xlabel(f'L{layer} H{head}', fontsize=20)

    fig.colorbar(cax, ax=axes.ravel().tolist(), shrink=0.95, aspect=40, pad=0.02)
    plt.savefig(file_name)
    plt.close()


# ----------------------test------------------------------
def greedy_decoder(model, enc_input, start_symbol_idx):
    """
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input [batchsize=1, src_len]
    :param start_symbol_idx: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_output, enc_self_attns = model.encoder(enc_input)
    dec_input = torch.zeros(1, 0).type_as(enc_input)
    terminal = False
    next_symbol_idx = start_symbol_idx
    while not terminal:
        dec_input = torch.cat([dec_input.detach(), torch.tensor([[next_symbol_idx]], dtype=enc_input.dtype).cuda()], -1)
        dec_output, dec_self_attns, dec_enc_attns = model.decoder(dec_input, enc_output,
                                                                  enc_input)  # [batchsize, tgt_len, d_model]
        dec_logit = model.projection(dec_output).squeeze(0)  # [tgt_len, tgt_vocab_size]
        next_word_idx = torch.argmax(dec_logit, dim=-1, keepdim=False)[-1]  # [tgt_len]
        next_symbol_idx = next_word_idx
        if next_symbol_idx == tgt_vocab["."]:
            terminal = True
            dec_input = torch.cat([dec_input.detach(), torch.tensor([[tgt_vocab["."]]], dtype=enc_input.dtype).cuda()],
                                  -1)
    return dec_input


# Test
enc_inputs, _, _ = next(iter(loader))
enc_inputs = enc_inputs.cuda()
for i in range(len(enc_inputs)):
    # enc_attns  list([1,n_heads, src_len, src_len])
    # dec_attns  list([1,n_heads, tgt_len, tgt_len])
    # cross_attns  list([1,n_heads, tgt_len, src_len])
    enc_input = enc_inputs[i]
    greedy_dec_input = greedy_decoder(model, enc_inputs[i].view(1, -1), start_symbol_idx=tgt_vocab["S"])
    predict, enc_attns, dec_attns, cross_attns = model(enc_inputs[i].view(1, -1), greedy_dec_input)
    predict = predict.squeeze(0).argmax(dim=-1, keepdim=False)  # [tgt_len]

    print(enc_input, '->', predict)
    print([src_idx2word[n.item()] for n in enc_input], '->', [tgt_idx2word[n.item()] for n in predict])

    # plot attention heatmap
    src_tokens = [src_idx2word[n.item()] for n in enc_input]
    tgt_tokens = [tgt_idx2word[n.item()] for n in greedy_dec_input[0]]

    plot_all_attention_weights(enc_attns, src_tokens, src_tokens, f'enc_attns_{i}.jpg')
    plot_all_attention_weights(dec_attns, tgt_tokens, tgt_tokens, f'dec_attns_{i}.jpg')
    plot_all_attention_weights(cross_attns, src_tokens, tgt_tokens, f'cross_attns_{i}.jpg')
    break
