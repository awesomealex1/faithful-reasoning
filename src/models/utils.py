"""
Copied from https://github.com/yuzhaouoe/KC-Mechanism/blob/master/utils.py
"""

import torch


def pad_and_merge_single_layer(attention_weights_list):
    """
    Pad and merge a list of attention weights tensors.

    Args:
        attention_weights_list (list): A list of attention weights tensors coming from different generation steps. Each tensor has shape (batch_size, num_heads, source_len, target_len).
        Notice that source and taget len can be different for each tensor.

    Returns:
        torch.Tensor: A tensor of attention weights for all times steps. The tensor has shape (batch_size, num_heads, cumulative_source_len, max_target_len).
    """

    # max_source_seq_len = max(map(lambda x: x.shape[2], attention_weights_list))
    max_target_seq_len = max(map(lambda x: x.shape[3], attention_weights_list))

    # print('source len, target len', max_source_seq_len, max_target_seq_len)
    # print('attention_weights_list:', attention_weights_list[0].shape, attention_weights_list[1].shape)

    # pad all attention weights so that they have the same target sequence length
    padded_attention_weights = [
        torch.nn.functional.pad(t, (0, max_target_seq_len - t.shape[3]), "constant", 0)
        for t in attention_weights_list
    ]

    # padded_attention_weights is a list of attention maps, each of shape (batch_size, num_heads, source_len, target_len)
    # we need to merge them such that the output is of shape (batch_size, num_heads, source_len, target_len)

    # print('padded_attention_weights:', padded_attention_weights[0].shape, padded_attention_weights[1].shape)

    # out = torch.stack(padded_attention_weights, dim=1)
    out = torch.cat(padded_attention_weights, dim=2)
    # print('out:', out.shape)
    return out


def merge_attention_weights(attention_weights_list):
    """
    Merge attention weights from multiple layers into a single list.

    Args:
        attention_weights_list (list): A list of attention weights. Each element in the list is a list of tensors representing the attention weights for each layer.
        [[l1, l2, l3], [l1, l2, l3], ...] where l1, l2, l3 are tensors of shape (batch_size, num_heads, seq_len, seq_len).
        Notice that seq_len can be different for each tensor.

    Returns:
        list: A list of merged attention weights from different time steps.
    """
    num_layers = len(attention_weights_list[0])
    attn_weights_by_layer = []

    # for now we have a list of time steps, each time step has a list of attention weights for each layer

    # first, we reorganize the attention weights by layer, so that that for each layer we have a list of attention weights at different time steps

    for layer in range(num_layers):
        attn_weights_by_layer.append([])
        for aw in attention_weights_list:
            attn_weights_by_layer[layer].append(aw[layer])

    # because the sequence length can be different for each time step, we need to pad the attention weights

    return [
        pad_and_merge_single_layer(attention_weights)
        for attention_weights in attn_weights_by_layer
    ]
