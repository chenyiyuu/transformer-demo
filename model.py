import math

import torch.nn as nn
import torch.nn.functional as F
import torch


def get_subsequence_attn_mask(seq):
    """
    Mask future output for target sequence. 1 denotes masked.
    :param seq: [batchsize, tgt_len]
    :return: [batchsize, tgt_len, tgt_len]
    """
    batch_size, tgt_len = seq.size()
    subsequence_attn_mask = torch.ones(size=(batch_size, tgt_len, tgt_len)).cuda()
    subsequence_attn_mask = torch.triu(subsequence_attn_mask, diagonal=1)
    return subsequence_attn_mask


def get_pad_attn_mask(seq_q, seq_k):
    """
    Get padding mask for attention. 1 denotes masked.
    :param seq_q: [batchsize, q_len]
    :param seq_k: [batchsize, k_len]
    :return: [batchsize, q_len, k_len]
    """
    batchsize, q_len, k_len = seq_q.size(0), seq_q.size(1), seq_k.size(1)
    pad_attn_mask = seq_k.eq(0).unsqueeze(1)  # [batchsize, 1, k_len]
    pad_attn_mask = pad_attn_mask.expand(batchsize, q_len, k_len)  # [batchsize, q_len, k_len]
    return pad_attn_mask


class PositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # PE(pos,2i) = sin(pos/10000**(2i/d_model))  PE(pos,2i+1) = cos(pos/10000**(2i/d_model))
        pe = torch.zeros(size=(max_len, d_model))  # [max_len, d_model]
        div_term = torch.exp(- torch.arange(0, d_model, 2) * math.log(10000.0) / d_model)  # [d_model//2]
        pos = torch.arange(max_len).unsqueeze(1)  # [max_len]
        pe[:, 0::2] = torch.sin(pos / div_term)
        pe[:, 1::2] = torch.cos(pos / div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add position encoding to the input.
        :param x: [batchsize, seq_len, d_model]
        :return: [batchsize, seq_len, d_model]
        """
        pe = self.pe[:, :x.size(1), :]
        output = self.dropout(x + pe)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.W_K = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)
        self.W_V = nn.Linear(self.d_model, self.n_heads * self.d_k, bias=False)

        self.W_O = nn.Linear(self.n_heads * self.d_k, self.d_model)
        self.ln = nn.LayerNorm(self.d_model)

    def scaled_dot_product_attention(self, Q, K, V, attn_mask):
        """
        Calculate the attention weights.
        :param Q: [batchsize, n_heads, q_len, d_k]
        :param K: [batchsize, n_heads, k_len, d_k]
        :param V: [batchsize, n_heads, k_len, d_k]
        :param attn_mask: [batchsize, seq_len, seq_len]
        :return: [batchsize, n_heads, q_len, d_k]
        """
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batchsize, n_heads, q_len, k_len]
        n_heads = Q.size(1)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # [batchsize, n_heads, q_len, k_len]
        attn.masked_fill(attn_mask, -1e9)  # masked = 1 transfer to negative infinite

        attn = F.softmax(attn, dim=-1)  # [batchsize, n_heads, q_len, k_len]
        output = torch.matmul(attn, V)  # # [batchsize, n_heads, q_len, d_k]
        return output, attn

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        Multi-head attention Layer.
        :param input_Q: [batchsize, q_len, d_model]
        :param input_K: [batchsize, k_len, d_model]
        :param input_V: [batchsize, k_len, d_model]
        :param attn_mask: [batchsize, q_len, k_len]
        :return: [batchsize, q_len, d_model]
        """
        x = input_Q

        batchsize = input_Q.size(0)
        # [batchsize, n_heads, q_len, d_k]
        Q = self.W_Q(input_Q).view(batchsize, -1, self.n_heads, self.d_k).transpose(1, 2)
        # [batchsize, n_heads, k_len, d_k]
        K = self.W_Q(input_K).view(batchsize, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_Q(input_V).view(batchsize, -1, self.n_heads, self.d_k).transpose(1, 2)

        # [batchsize, n_heads, q_len, d_k]
        output, attn = self.scaled_dot_product_attention(Q, K, V, attn_mask)

        # [batchsize, q_len, d_model(n_heads*d_k)]
        output = output.transpose(1, 2).contiguous().view(batchsize, -1, self.n_heads * self.d_k)
        output = self.W_O(output)  # [batchsize, q_len, d_model(n_heads*d_k)]

        output = self.ln(x + output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
        self.ln = nn.LayerNorm(self.d_model)

    def forward(self, x):
        """
        PositionwiseFeedForward.
        :param x: [batchsize, seq_len, d_model]
        :return: [batchsize, seq_len, d_model]
        """
        # Feed Forward
        output = self.W1(x)
        output = F.relu(output)
        output = self.W2(output)

        # Add & Norm
        output = self.ln(x + output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        """
        Encoder layer.
        :param enc_inputs: [batchsize, src_len, d_model]]
        :param enc_self_attn_mask: [batchsize, src_len, src_len]
        :return: enc_outputs: [batchsize, src_len, d_model]
        """

        enc_outputs, enc_attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)

        enc_outputs = self.pos_ffn(enc_outputs)

        return enc_outputs, enc_attn


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(DecoderLayer, self).__init__()

        self.dec_self_attn = MultiHeadAttention(d_model, n_heads)
        self.dec_enc_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_ff)

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        Decoder layer.
        :param dec_inputs: [batchsize, tgt_len, d_model]
        :param enc_outputs: [batchsize, src_len, d_model]
        :param dec_self_attn_mask: [batchsize, tge_len, tgt_len]
        :param dec_enc_attn_mask: [batchsize, tgt_len, src_len]
        :return: [batchsize, tgt_len, d_model]
        """
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)

        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)

        dec_outputs = self.pos_ffn(dec_outputs)

        return dec_outputs, dec_self_attn, dec_enc_attn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, n_layers, d_model, n_heads, d_ff):
        super(Encoder, self).__init__()

        self.input_embedding = nn.Embedding(src_vocab_size, d_model)
        self.position_embedding = PositionEncoding(d_model)

        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        Encoder.
        :param enc_inputs: [batchsize, src_len]
        :return: [batchsize, src_len, d_model]
        """

        enc_outputs = self.input_embedding(enc_inputs)
        enc_outputs = self.position_embedding(enc_outputs)

        enc_self_attn_mask = get_pad_attn_mask(enc_inputs, enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)

        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, n_layers, d_model, n_heads, d_ff):
        super(Decoder, self).__init__()

        self.output_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.position_embedding = PositionEncoding(d_model)

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_outputs, enc_inputs):
        """
        Decoder.
        :param dec_inputs: [batchsize, tgt_len]
        :param enc_outputs: [batchsize, src_len, d_model]
        :param enc_inputs:  [batchsize, src_len]
        :return: [batchsize, tgt_len, d_model]
        """
        dec_outputs = self.output_embedding(dec_inputs)
        dec_outputs = self.position_embedding(dec_outputs)

        dec_self_attn_mask_subsequence = get_subsequence_attn_mask(dec_inputs)
        dec_self_attn_mask_pad = get_pad_attn_mask(dec_inputs, dec_inputs)
        dec_self_attn_mask = torch.gt((dec_self_attn_mask_subsequence + dec_self_attn_mask_pad), 0)

        dec_enc_attn_mask = get_pad_attn_mask(dec_inputs, enc_inputs)

        dec_self_attns, dec_enc_attns = [], []

        for layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, n_layers, d_model, n_heads, d_ff):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, n_layers, d_model, n_heads, d_ff)
        self.decoder = Decoder(tgt_vocab_size, n_layers, d_model, n_heads, d_ff)

        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        """
        Transformer.
        :param enc_inputs: [batchsize, src_len]
        :param dec_inputs: [batchsize, tgt_len]
        :return: [batchsize, tgt_len, tgt_vocab_size]
        """
        enc_outputs, enc_self_attns = self.encoder(enc_inputs) # [batchsize, src_len, d_model]

        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_outputs, enc_inputs) # [batchsize, tgt_len, d_model]

        dec_logits = self.projection(dec_outputs)  # [batchsize, tgt_len, tgt_vocab_size]

        return dec_logits, enc_self_attns, dec_self_attns, dec_enc_attns
