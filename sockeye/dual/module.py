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
from .. import lmscore

from ..model import ModelConfig 
from ..builder import ModelBuilder

import pdb
import copy
import mxnet as mx
from typing import AnyStr, List, Optional

import inspect

from .. import beam_search

from .config import DualConfig
from . import outop

def PrintFrame():
    """
    print grandparent's path , for debug
    """
    callerframerecord = inspect.stack()[2]    # 0: this line, 1: parent line, 2: grandparent line
    frame = callerframerecord[0]
    info = inspect.getframeinfo(frame)
    return info.lineno


def debug_shape(sym):
    """
    used to debug shape of symbol
    """
    lineno = PrintFrame() 

    if int==type(sym):
        print(lineno, sym)
        return 

    inputs = sym.list_inputs()
    fsrc = 0
    ftgt = 0
    flab = 0

    srcshape=(30,50)
    tgtshape=(30,50)

    for item in inputs:
        if 'source'==item:
            fsrc = 1
        elif 'target'==1:
            ftgt = 1
        elif 'target_label'==1:
            flab = 1

    if 1==fsrc:
        if 0==ftgt:
            print(lineno, sym.name, sym.infer_shape(source=srcshape))
        else:
            print(lineno, sym.name, sym.infer_shape(source=srcshape, target=tgtshape))
    elif 1==ftgt:
        print(lineno, sym.name, sym.infer_shape(target=tgtshape))


class DualEncoderDecoderBuilder(ModelBuilder):
    """
    dual translate model with 
    """
    def __init__(self, context: List[mx.context.Context], config: ModelConfig,
            train_iter: data_io.BaseParallelSampleIter, logger) -> None:

        super().__init__(context, config, train_iter, logger)
        self.prefix = 'dual_'
        self.config_all = config
        self.config = config.config_dual
 
        self.target_lengths = mx.sym.full(shape=(self.config.batch_size,), val=config.max_seq_len_target)

        self.labels = mx.sym.Variable(C.TARGET_LABEL_NAME)



    ##
    # @brief a->b in dual-learning model, do a beam-search in training
    #
    # @return 
    def beam_decode(self, model, source_encoded, source_encoded_seq_len, 
            target_seq_len, beam_size):
        target_vocab_size = self.config.vocab_target_size

        # [batch_size, 1, source_seq_len, num_hidden]
        # -> [batch_size, beam_size, source_seq_len, encode_hidden_len]
        # -> [batch_size*beam_size, source_seq_len, encode_hidden_len]
        repeat_encoded = mx.sym.expand_dims(source_encoded, axis=1, name="soure_encode_expand_dim")
        repeat_encoded = mx.sym.repeat(repeat_encoded, repeats=beam_size, axis=1, name="repeat_encoded")
        repeat_encoded = mx.sym.reshape(repeat_encoded, shape=(-3,0,0), name="repeat_encoded_reshape")

        # [batch_size, 1] -> [batch_size, beam_size] -> [batc_size*beam_size,]
        repeat_encoded_lengths = mx.sym.expand_dims(self.source_lengths, axis=1, name="source_encode_length_expand_dim")
        repeat_encoded_lengths = mx.sym.repeat(repeat_encoded_lengths, repeats=beam_size, axis=1, name="repeat_encoded_length")
        repeat_encoded_lengths = mx.sym.reshape(repeat_encoded_lengths, shape=(-1,), name="repeat_encoded_length_reshape")

        repeat_encoded_tmajor = mx.sym.swapaxes(repeat_encoded, dim1=0, dim2=1, name="repeat_encoded_tmajor")  
        state = model.decoder.get_initial_state(repeat_encoded_tmajor, repeat_encoded_lengths)

        # TODO 
        # this is is MUST appear to make gradient flow back, YYYYYY?????
        repeat_for_att = mx.sym.reshape(repeat_encoded, shape=(0,0,0), name="repeat_for_att")
        attention_func = model.decoder.attention.on(repeat_for_att, repeat_encoded_lengths, source_encoded_seq_len)
        attention_state = model.decoder.attention.get_initial_state(repeat_encoded_lengths, source_encoded_seq_len)

        # [batch_size, 1] -> [batch_size, beam_size]
        target_prev = mx.sym.zeros(shape=(self.config.batch_size, beam_size), name='target_prev-init')

        target_row_all = []
        target_col_all = []
        target_prob_all = []

        # decode time step
        for t in range(target_seq_len):
            # 001. get previous target embedding
            # [batch_size, beam_size, num_embed]
            target_embed_prev,_,_ = model.embedding_target.encode(target_prev, self.target_lengths, target_seq_len)

            # [batch_size*beam_size, num_embed]
            target_embed_prev = mx.sym.reshape(target_embed_prev, shape=(-3,0), name='target_embed_prev-%d'%t)

            # 002. 1-step forward
            #
            # [batch_size*beam_size, num_hidden]
            #(target_decoded, attention_probs, states) = model.decoder.decode_step(t, target_embed_prev, source_encoded_seq_len, *state)

            state, attention_state = model.decoder._step(target_embed_prev, state, attention_func, attention_state, t)
            target_decoded = state.hidden

            # 003. output projection
            #
            # should we do softmax here???
            #
            # [batch_size*beam_size, target_vocab_size]
            pred = model.output_layer(target_decoded) 

            # 004. length penalty
            # TODO

            # 005. get top-k prediction
            # -> [batch_size, beam_size, target_vocab_size]
            # -> [batch_size, beam_size*target_vocab_size]
            pred = mx.sym.reshape(pred, shape=(-4, -1, beam_size, 0))
            pred = mx.sym.reshape(pred, shape=(0,-3), name='prediction-%d'%t) 

            # [batch_size, beam_size]
            topkval,topkpos = mx.sym.topk(pred, k=beam_size, ret_typ="both", axis=-1, name="topk-%d"%t)
            topk_pos_row = topkpos/target_vocab_size
            topk_pos_col = topkpos%target_vocab_size

            # 005. save result for loss compute [1, batch_size, beam_size]
            target_row_all.append(mx.sym.expand_dims(topk_pos_row, axis=0))
            target_col_all.append(mx.sym.expand_dims(topk_pos_col, axis=0))
            target_prob_all.append(mx.sym.expand_dims(topkval, axis=0, name="tok-%d_prob" % t))

            # 006. update target_prev
            # [batch_size, beam_size] 
            target_prev = mx.sym.reshape(topk_pos_col, shape=(0,0), name='target_prev-%d'%t)

        # a list of length target_seq_len, contain items in size [1, batch_size, beam_size]
        # -> [target_seq_len, batch_size, beam_size]
        target_rows = mx.sym.concat(*target_row_all, dim=0, name="beam_search_concat_rows") 
        target_cols = mx.sym.concat(*target_col_all, dim=0, name="beam_search_concat_cols")
        target_prob = mx.sym.concat(*target_prob_all, dim=0, name="beam_search_concat_prob")

        # [target_seq_len, batch_size, beam_size]
        path, prob = mx.sym.Custom(rows=target_rows, cols=target_cols, probs=target_prob, name='beam_search', op_type='beam_search') 

        return path, prob
    

    def forward(self, model, bucket):
        """
        do forward translate of dual-translate 

        PARAMETERS:
            - model: model factory used to generate symbo
            - bucket
        """
        # source_seq_len: max input utterance length
        # target_seq_len: max output utterance length
        source_seq_len, target_seq_len = bucket

        # source embedding
        #
        # source_embed_length: same as source_length
        # source_embed_seq_len: same as source_seq_len
        # source_embed: [batch_size, source_seq_len, source_hidden_size]
        (source_embed,
         source_embed_length,
         source_embed_seq_len) = model.embedding_source.encode(self.source, self.source_lengths, source_seq_len)

        # encoder
        #
        # source_encoded_length: [batch_size, ], same as source_embed_length
        # source_encoded_seq_len: same as source_embed_seq_len, max length of input utterances
        # source_encoded: [source_seq_len, batch_size, num_hidden]
        (source_encoded,
         source_encoded_lengths,
         source_encoded_seq_len) = model.encoder.encode(source_embed,
                                                       source_embed_length,
                                                       source_embed_seq_len)
        source_encoded = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1, name="batch_major_source_encoded")

        # beam_decoded: [batch_size, target_seq_len, target_vocab_size]
        # beam_path:    [target_seq_len, batch_size, beam_size]
        beam_path, path_prob = self.beam_decode(model, source_encoded, source_encoded_seq_len, 
            target_seq_len, self.config.beam_size)

        return beam_path, path_prob


    def backward(self, model, bucket, beam_out):
        '''
        PARAMETERS:
            - beam_path:    [batch_size, target_seq_len, beam_size]
        '''
        target_seq_len, source_seq_len = bucket
        beam_size = self.config.beam_size

        # [target_seq_len, batch_size, beam_size]
        # -> [target_seq_len, batch_size*beam_size]
        # -> [batch_size*beam_size, target_seq_len]
        inputs = mx.sym.reshape(beam_out, shape=(0, -3))
        inputs = mx.sym.swapaxes(inputs, dim1=0, dim2=1)
        inputs = mx.sym.BlockGrad(inputs)

        # [batch_size*beam_size,]
        source_lengths = utils.compute_lengths(inputs)

        # [batch_size, 1] -> [batch_size, beam_size] -> [batc_size*beam_size,]
        target_lengths = mx.sym.expand_dims(self.source_lengths, axis=1)
        target_lengths = mx.sym.repeat(target_lengths, repeats=beam_size, axis=1)
        target_lengths = mx.sym.reshape(target_lengths, shape=(-1,))

        # [batch_size, source_seq_len] -> [batch_size, 1, source_seq_len]
        # -> [batch_size, beam_size, source_seq_len] 
        # -> [batc_size*beam_size, source_seq_len]
        target = mx.sym.expand_dims(self.source, axis=1)
        target = mx.sym.repeat(target, repeats=beam_size, axis=1)
        target = mx.sym.reshape(target, shape=(-3, 0))

        # inputs embedding
        # [batch_size*beam_size, target_seq_len, num_embed]
        (source_embed,
         source_embed_length,
         source_embed_seq_len) = model.embedding_source.encode(inputs, source_lengths, source_seq_len)

        # target embedding
        # [batch_size*beam_size, source_seq_len, num_embed]
        (target_embed,
         target_embed_length,
         target_embed_seq_len) = model.embedding_target.encode(target, target_lengths, target_seq_len)

        # encoder
        # source_encoded: (target_encoded_length, batch_size*beam_size, encoder_depth)
        (source_encoded,
         source_encoded_length,
         source_encoded_seq_len) = model.encoder.encode(source_embed, source_embed_length, source_embed_seq_len)

        # decoder
        # target_decoded: (batch_size*beam_size, source_len, decoder_depth)
        target_decoded = model.decoder.decode_sequence(source_encoded, source_encoded_length, source_encoded_seq_len,
                                                      target_embed, target_embed_length, target_embed_seq_len)

        # output layer
        # logits: (batch_size*beam_size, source_seq_len, target_vocab_size)
        pred = model.output_layer(target_decoded)

        return pred


    def sym_gen_predict(self, bucket, prefix=""):
        """
        Returns a (grouped) loss symbol given source & target input lengths.
        Also returns data and label names for the BucketingModule.
        """
        source_seq_len,target_seq_len = bucket

        model_forward = model.SockeyeModel(self.config_all, "%sf_"%self.prefix)
        model_forward._build_model_components()

        model_backward = model.SockeyeModel(self.config_all, "%sb_"%self.prefix)
        model_backward._build_model_components()

        # [target_seq_len, batch_size, beam_size]
        forward_path, path_prob = self.forward(model_forward, bucket)

        # [batch_size*beam_size, source_seq_len, source_vocab_size]
        backward_decoded = self.backward(model_backward, bucket, forward_path)

        return [forward_path, path_prob, backward_decoded, source_seq_len]


    def get_loss(self, logits):
        # forward_path: [batch_size, target_seq_len, beam_size]
        # path_prob: [batch_size, target_seq_len, beam_size]
        # backward_logits: [batch_size*beam_size, source_seq_len, source_vocab_size]
        forward_path, path_prob, backward_logits, source_seq_len = logits

        # STEP 1. importance sampling of AB output probability for each beam sentences
        #
        # L1 normalization
        #
        # [target_seq_len, batch_size, beam_size] 
        # -> [batch_size, beam_size]
        # -> [batch_size, 1]
        path_prob = path_prob + 1e-8
        path_logits = -mx.sym.log(path_prob, name='path_prob_logits')
        path_prob = mx.sym.sum(path_prob, axis=0, name='path_prob_sum_on_time') 
        #prob_sum = mx.sym.sum(path_prob, axis=1, keepdims=True, name='path_prob_sum_for_norm')

        # [batch_size, beam_size] 
        # -> [batch_size*beam_size]
        #path_prob = mx.sym.broadcast_div(path_prob, prob_sum, name='path_prob_normed')  
        path_prob = mx.sym.reshape(path_prob, shape=(-1,), name='path_prob_normed_reshaped')

        # STEP 2. language model score 
        # [target_seq_len, batch_size, beam_size]
        # -> [target_seq_len, batch_size*beam_size]
        # -> [batch_size*beam_size, target_seq_len]
        # -> [batch_size*beam_size, target_seq_len]: -log(p(y|model))
        # -> [batch_size*beam_size]
        forward_path = mx.sym.reshape(forward_path, shape=(0, -3)) 
        forward_path = mx.sym.swapaxes(forward_path, dim1=0, dim2=1) 
        lm_logits = mx.sym.Custom(data=forward_path, op_type='lm_score', 
                prefix=self.config.lm_prefix, epoch=self.config.lm_epoch, pad=C.PAD_ID,
                devid=self.config.lm_device_ids)
        lm_score = mx.sym.sum(lm_logits, axis=-1)
        lm_score = mx.sym.BlockGrad(lm_score)

        # STEP 3. BA model    
        # [batch_size, source_seq_len]
        # -> [batch_size, 1, source_seq_len]
        # -> [batch_size, beam_size, source_seq_len]
        # -> [batch_size*beam_size, source_seq_len]
        label = mx.sym.expand_dims(self.labels, axis=1)
        label = mx.sym.repeat(label, repeats=self.config.beam_size, axis=1)
        label = mx.sym.reshape(label, shape=(-3,0))
        ignore = (C.PAD_ID==label)

        # [batch_size*beam_size, source_seq_len, source_vocab_size]
        # -> [batch_size*beam_size, source_seq_len]
        # -> [batch_size*beam_size, source_seq_len]
        # -> [batch_size*beam_size, source_seq_len]
        # -> [batch_size*beam_size,]
        backward_logits = mx.sym.softmax(backward_logits)
        backward_logits = mx.sym.pick(backward_logits, label)
        backward_logits = (1-ignore)*backward_logits + ignore
        backward_logits = -mx.sym.log(backward_logits+1e-8)
        backward_score = mx.sym.sum(backward_logits, axis=-1)

        # [batch_size*beam_size]
        loss = mx.sym.Custom(lm_score=lm_score, path_prob=path_prob, backward_score=backward_score, target=self.target,
                 op_type='dual_output', alpha=self.config.alpha, scale=self.config.forward_gradient_scale, 
                 beam_size=self.config.beam_size, name='dual_output')
        return [loss]
