#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import sys
import mxnet as mx
from . import dual_model


##
# @brief a->b in dual-learning model, do a beam-search in training
#
# @param source_encoded: hidden return by encoder, [batch_size, source_seq_len, encoded_hidden_len] 
# @param source_encoded_lengths: length of each utterance in current batch, [batch_size, 1] 
#
# @return 
def beam_decode_model(model, beam_size,
        source_encoded, source_encoded_lengths, source_encoded_seq_len, 
        target, target_embed, target_lengths, target_seq_len):
    prefix = "dual"
    target_vocab_size = model.config.vocab_target_size
    target_embed_split = mx.sym.split(target_embed, num_outputs=target_seq_len, axis=1)

    # [batch_size, 1] -> [batch_size, beam_size] -> [batc_size*beam_size,]
    source_encoded_lengths = mx.sym.expand_dims(source_encoded_lengths, axis=1)
    source_encoded_lengths = mx.sym.repeat(source_encoded_lengths, repeats=beam_size, axis=1)
    source_encoded_lengths = mx.sym.reshape(source_encoded_lengths, shape=(-1,))

    # [batch_size, 1, source_seq_len, encoded_hidden_len]
    # -> [batch_size, beam_size, source_seq_len, encode_hidden_len]
    # -> [batch_size*beam_size, source_seq_len, encode_hidden_len]
    repeat_encoded = mx.sym.expand_dims(source_encoded, axis=1)
    repeat_encoded = mx.sym.repeat(repeat_encoded, repeats=beam_size, axis=1)
    repeat_encoded = mx.sym.reshape(repeat_encoded, shape=(-3,0,0))

    # state0
    #   - source_encoded: 
    #   - dynamic_source 
    #   - source_encoded_lengths 
    #   - hidden 
    #   - layer_states
    state = model.decoder.init_states(repeat_encoded, source_encoded_lengths, source_encoded_seq_len)

    # [batch_size, 1] -> [batch_size, beam_size]
    target_prev = mx.sym.slice_axis(target, axis=-1, begin=0, end=1)
    target_prev = mx.sym.zeros_like(target_prev)
    target_prev = mx.sym.repeat(target_prev, repeats=beam_size, axis=-1)

    target_row_all = []
    target_col_all = []
    pred_4loss_all = []   # just for debug, keeping the original loss compute, while can testing top-k in forward compute

    # decode time step
    for t in range(target_seq_len):
        # 001. get previous target embedding
        # [batch_size, beam_size, num_embed]
        target_embed_prev,_,_ = model.embedding_target.encode(target_prev, target_lengths, target_seq_len)

        # just for debugging
        # target_embed_split[t]: [batch_size, 1, num_embed]
        # target_embed_prev: [batch_size, beam_size, num_embed]
        target_embed_prev = mx.sym.broadcast_add(lhs=target_embed_prev, rhs=target_embed_split[t])

        # [batch_size*beam_size, num_embed]
        target_embed_prev = mx.sym.reshape(target_embed_prev, shape=(-3,0))

        # 002. 1-step forward
        #
        # [batch_size*beam_size, num_hidden]
        (target_decoded, attention_probs, states) = model.decoder.decode_step(t, target_embed_prev, source_encoded_seq_len, *state)

        # 003. output projection
        #
        # should we do softmax here???
        #
        # [batch_size*beam_size, target_vocab_size]
        pred = model.output_layer(target_decoded) 

        # just for debug, keep current loss computation function not complain for new model
        #
        # [batch_size, beam_size, target_vocab_size]
        # [batch_size, 1, target_vocab_size]
        pred_split = mx.sym.reshape(pred, shape=(-4, -1, beam_size, target_vocab_size))
        pred_split = mx.sym.split(pred_split, num_outputs=beam_size, axis=1)
        pred_4loss_all.append(pred_split[0])

        # 004. length penalty
        # TODO

        # 005. get top-k prediction
        # -> [batch_size, beam_size, target_vocab_size]
        # -> [batch_size, beam_size*target_vocab_size]
        pred = mx.sym.reshape(pred,shape=(-4, -1, beam_size, 0))
        pred = mx.sym.reshape(pred, shape=(0,-3)) 

        # [batch_size, beam_size]
        topkval,topkpos = mx.sym.topk(pred, k=beam_size, ret_typ="both", axis=-1)
        topk_pos_row = mx.sym.floor(topkpos/target_vocab_size)
        topk_pos_col = topkpos%target_vocab_size

        # 005. save result for loss compute
        target_row_all.append(topk_pos_row)
        target_col_all.append(topk_pos_col)

        # 006. update target_prev
        # [batch_size, beam_size] 
        target_prev = topk_pos_col

    # [batch_size, target_seq_len, target_vocab_size]
    pred_4loss = mx.sym.concat(*pred_4loss_all, dim=1, name="%starget_4loss_concat" % prefix)

    # [batch_size, target_seq_len, beam_size]
    path = get_path_from_beam(target_row_all, target_col_all, target_seq_len) 

    return pred_4loss, path



##
# @brief get path from beam search result 
#
# @param rows, a list whoes length is target_seq_len, contain items in size [batch_size,beam_size]
# @param cols, same as row
#
# @return 
def get_path_from_beam(rows, cols, target_seq_len):
    final_path = []

    # [batch_size, beam_size]
    curpos = cols[target_seq_len-1] 
    currow = rows[target_seq_len-1]

    for i in range(target_seq_len-2, -1, -1):
        # [batch_size, 1, beam_size], for final concat
        curpos = mx.sym.expand_dims(curpos, axis=1) 
        final_path.insert(0, curpos) 

        # [batch_size, beam_size]
        curpos = mx.sym.pick(cols[i], currow)
        currow = mx.sym.pick(rows[i], currow)

    # list of [batch_size, beam_size], which length is target_seq_len
    final_path.insert(0, curpos)  

    # [batch_size, target_seq_len, beam_size]
    return mx.sym.concat(*final_path, dim=1)
