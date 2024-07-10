import torch
import torch.utils.data as data
import dataset
from model import Transformer

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

# target vocabulary
tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'coke': 5, 'S': 6, 'E': 7, '.': 8}
tgt_vocab_size = len(tgt_vocab)
idx2word = {i: w for w, i in tgt_vocab.items()}

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

# -------------------loss---------------------------
criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)

# -------------------train---------------------------
for epoch in range(100):
    losses = []
    for batch in loader:
        '''
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        '''
        enc_inputs, dec_inputs, dec_outputs = [x.to('cuda') for x in batch]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
        # outputs : [batch_size, tgt_len, tgt_vocab_size]
        loss = criterion(outputs.view(-1, outputs.size(-1)), dec_outputs.view(-1))
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(sum(losses) / len(losses)))

torch.save(model.state_dict(), 'transformer.pth')
