#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(auimoviki@gmail.com)

from .. import loss
from .. import model
from .. import utils
from .. import data_io
from .. import constants as C

from .builder import ModelBuilder

import copy
import mxnet as mx
from typing import AnyStr, List, Optional

class TopKEncoderDecoderBuilder(ModelBuilder):
    def __init__(self, context: List[mx.context.Context],
            config: model.ModelConfig,
            train_iter: data_io.BaseParallelSampleIter, logger, k) -> None:
        super().__init__(context, config, train_iter, logger)
        self.beam_size = k
        self.prefix = 'topk_'


    ##
    # @brief a->b in dual-learning model, do a beam-search in training
    #
    # @return 
    def _beam_decode(self, model, source_encoded, source_encoded_seq_len, 
            target_embed, target_seq_len):
        beam_size = self.beam_size 

        target_vocab_size = self.config.vocab_target_size

        # list of [batch_size, 1, num_embed], which length is target_seq_len
        target_embed_split = mx.sym.split(target_embed, num_outputs=target_seq_len, axis=1)

        # [batch_size, 1, source_seq_len, encoded_hidden_len]
        # -> [batch_size, beam_size, source_seq_len, encode_hidden_len]
        # -> [batch_size*beam_size, source_seq_len, encode_hidden_len]
        repeat_encoded = mx.sym.expand_dims(source_encoded, axis=1)
        repeat_encoded = mx.sym.repeat(repeat_encoded, repeats=beam_size, axis=1)
        repeat_encoded = mx.sym.reshape(repeat_encoded, shape=(-3,0,0))

        # [batch_size, 1] -> [batch_size, beam_size] -> [batc_size*beam_size,]
        repeat_encoded_lengths = mx.sym.expand_dims(self.source_lengths, axis=1)
        repeat_encoded_lengths = mx.sym.repeat(repeat_encoded_lengths, repeats=beam_size, axis=1)
        repeat_encoded_lengths = mx.sym.reshape(repeat_encoded_lengths, shape=(-1,))

        state = model.decoder.init_states(repeat_encoded, repeat_encoded_lengths, source_encoded_seq_len)

        # [batch_size, 1] -> [batch_size, beam_size]
        target_prev = mx.sym.slice_axis(self.target, axis=-1, begin=0, end=1)
        target_prev = mx.sym.zeros_like(target_prev)
        target_prev = mx.sym.repeat(target_prev, repeats=beam_size, axis=-1)

        target_row_all = []
        target_col_all = []
        pred_4loss_all = []   # just for debug, keeping the original loss compute, while can testing top-k in forward compute

        # decode time step
        for t in range(target_seq_len):
            # 001. get previous target embedding
            # [batch_size, beam_size, num_embed]
            target_embed_prev,_,_ = model.embedding_target.encode(target_prev, self.target_lengths, target_seq_len)

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
        pred_4loss = mx.sym.concat(*pred_4loss_all, dim=1)

        # [batch_size, target_seq_len, beam_size]
        path = self._get_path_from_beam(target_row_all, target_col_all, target_seq_len) 

        return pred_4loss, path

    ##
    # @brief get path from beam search result 
    #
    # @param rows, a list whoes length is target_seq_len, contain items in size [batch_size,beam_size]
    # @param cols, same as row
    #
    # @return 
    def _get_path_from_beam(self, rows, cols, target_seq_len):
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


    def sym_gen_predict(self, seq_lens, prefix=""):
        """
        Returns a (grouped) loss symbol given source & target input lengths.
        Also returns data and label names for the BucketingModule.
        """
        module_factory = model.SockeyeModel(self.config, prefix)
        module_factory._build_model_components()

        # source_seq_len: max input utterance length
        # target_seq_len: max output utterance length
        source_seq_len, target_seq_len = seq_lens

        # source embedding
        #
        # source_embed_length: same as source_length
        # source_embed_seq_len: same as source_seq_len
        # source_embed: [batch_size, source_seq_len, source_hidden_size]
        (source_embed,
         source_embed_length,
         source_embed_seq_len) = module_factory.embedding_source.encode(self.source, self.source_lengths, source_seq_len)

        # target embedding
        # 
        # target_embed: [batch_size, target_seq_len, target_hidden_size] 
        (target_embed,
         target_embed_lengths,
         target_embed_seq_len) = module_factory.embedding_target.encode(self.target, self.target_lengths, target_seq_len)

        # encoder
        #
        # source_encoded_length: [batch_size, ], same as source_embed_length
        # source_encoded_seq_len: same as source_embed_seq_len, max length of input utterances
        # source_encoded: [source_seq_len, batch_size, encoded_hidden_len]
        (source_encoded,
         source_encoded_lengths,
         source_encoded_seq_len) = module_factory.encoder.encode(source_embed,
                                                       source_embed_length,
                                                       source_embed_seq_len)
        source_encoded = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1)

        # beam_decoded: [batch_size, target_seq_len, target_vocab_size]
        # beam_path:    [batch_size, target_seq_len, beam_size]

        beam_decoded, beam_path = self._beam_decode(module_factory, source_encoded, source_encoded_seq_len, 
            target_embed, target_embed_seq_len)

        # do B->A translate
        # -> [batch_size, beam_size, target_seq_len]
        # -> [batch_size*beam_size, target_seq_len]
        #beam_path = mx.sym.swapaxes(beam_path, dim1=1, dim2=2)
        #beam_path = mx.sym.reshape(beam_path, shape=(-3, 0))

        # compute loss
        # [batch_size*target_seq_len, target_vocab_size]
        logits = mx.sym.reshape(beam_decoded, shape=(-3, 0))

        return logits
