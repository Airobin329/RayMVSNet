# Attention layers for tensor shape (B, N, C).
# Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).

import torch
import torch.nn as nn
import torch.nn.functional as F

# from vision3d.utils.torch_utils import get_activation, get_dropout


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_head):
        super(MultiHeadAttention, self).__init__()
        if d_model % num_head != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_head` ({}).'.format(d_model, num_head))

        self.d_model = d_model
        self.num_head = num_head
        self.d_model_per_head = d_model // num_head

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

    def _transpose_for_scores(self, x):
        x = x.view(x.shape[0], x.shape[1], self.num_head, self.d_model_per_head)
        x = x.permute(0, 2, 1, 3)
        return x

    def forward(self, input_q, input_k, input_v, key_masks=None):
        '''
        :param input_q: torch.Tensor (B, N, C)
        :param input_k: torch.Tensor (B, M, C)
        :param input_v: torch.Tensor (B, M, C)
        :param key_masks: torch.Tensor (B, M), -inf if masked, 0 otherwise
        :return: hidden_states: torch.Tensor (B, C, N)
        '''
        q = self.proj_q(input_q)
        k = self.proj_k(input_k)
        v = self.proj_v(input_v)

        q = self._transpose_for_scores(q)
        k = self._transpose_for_scores(k)
        v = self._transpose_for_scores(v)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / self.d_model_per_head ** 0.5
        if key_masks is not None:
            attention_scores = attention_scores + key_masks.unsqueeze(1)
        attention_scores = F.softmax(attention_scores, dim=-1)
        # from IPython import embed
        # embed()

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = hidden_states.permute(0, 2, 1, 3).contiguous()
        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], self.d_model)

        return hidden_states


class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_head):
        super(AttentionLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_head)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_states, memory_states, memory_masks=None):
        hidden_states = self.attention(input_states, memory_states, memory_states, key_masks=memory_masks)
        hidden_states = self.linear(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states


class AttentionOutput(nn.Module):
    def __init__(self, d_model):
        super(AttentionOutput, self).__init__()
        '''
        self.expand = nn.Linear(d_model, d_model * 2)
        self.activation = nn.ReLU()
        self.squeeze = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        '''

        self.activation = nn.ReLU()
        self.squeeze = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)        

    def forward(self, input_states):
        # hidden_states = self.expand(input_states)
        # hidden_states = self.activation(hidden_states)
        hidden_states = self.activation(input_states)
        hidden_states = self.squeeze(hidden_states)
        output_states = self.norm(input_states + hidden_states)
        return output_states


class TransformerLayer(nn.Module):
    def __init__(self, d_model, num_head):
        super(TransformerLayer, self).__init__()
        self.attention = AttentionLayer(d_model, num_head)
        self.output = AttentionOutput(d_model)

    def forward(self, input_states, memory_states, memory_masks=None):
        hidden_states = self.attention(input_states, memory_states, memory_masks=memory_masks)
        output_states = self.output(hidden_states)
        # output_states = hidden_states
        return output_states


class AttentionalGNN(nn.Module):
    def __init__(self, blocks, d_model, num_head, activation_fn='gelu'):
        super(AttentionalGNN, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            if block not in ['self', 'cross']:
                raise ValueError('Unsupported block type "{}" in `AttentionalGNN`.'.format(block))
            layers.append(TransformerLayer(d_model, num_head, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)

    def forward(self, feats0, feats1, embeddings0=None, embeddings1=None, masks0=None, masks1=None):
        for i, block in enumerate(self.blocks):
            if embeddings0 is not None:
                feats0 = feats0 + embeddings0
            if embeddings1 is not None:
                feats1 = feats1 + embeddings1
            if block == 'self':
                feats0 = self.layers[i](feats0, feats0, memory_masks=masks0)
                feats1 = self.layers[i](feats1, feats1, memory_masks=masks1)
            else:
                feats0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1 = self.layers[i](feats1, feats0, memory_masks=masks0)
        return feats0, feats1
