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

    srcshape=(20,40)
    tgtshape=(20,36)

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
 
        self.labels = mx.sym.Variable(C.TARGET_LABEL_NAME)


    ##
    # @brief get path from beam search result 
    #
    # @param rows, a list whoes length is target_seq_len, contain items in size [batch_size,beam_size]
    # @param cols, same as row
    # @param probs, probability of each timestamp, same shape as row
    # @param target_seq_len, the length of rows and cols 
    #
    # @return valid path generate from rows and cols, of size [batch_size, target_seq_len, beam_size]
    def _get_path_from_beam(self, rows, cols, probs, target_seq_len):
        final_path = []
        final_prob = []

        # [batch_size, beam_size]
        batch_size = self.config.batch_size
        beam_size = self.config.beam_size

        # list of [1,beam_size], of length batch_size
        precol = mx.sym.split(cols[target_seq_len-1], axis=0, num_outputs=batch_size)
        preprob = mx.sym.split(probs[target_seq_len-1], axis=0, num_outputs=batch_size)

        # list of [beam_size], of length batch_size
        prerow = mx.sym.split(rows[target_seq_len-1], axis=0, num_outputs=batch_size, squeeze_axis=1)

        for i in range(target_seq_len-2, -1, -1):
            final_path.extend(precol)
            final_prob.extend(preprob)

            # list of [1,beam_size], of length batch_size
            curcol = mx.sym.split(cols[i], axis=0, num_outputs=batch_size)  
            currow = mx.sym.split(rows[i], axis=0, num_outputs=batch_size)
            curprob = mx.sym.split(probs[i], axis=0, num_outputs=batch_size)

            tmpcol = []
            tmprow = []
            tmpprob = []
            for j in range(0, batch_size):
                # [1, beam_size] -> [beam_size, beam_size]
                # -> [beam_size,]
                # -> [1, beam_size]
                tmp = mx.sym.repeat(curcol[j], repeats=beam_size, axis=0) 
                tmp = mx.sym.pick(tmp, prerow[j])
                tmp = mx.sym.reshape(tmp, shape=(-4,1,-1))
                tmpcol.append(tmp)

                # [1, beam_size] -> [beam_size, beam_size]
                # -> [beam_size,]
                # -> [1, beam_size]
                tmp = mx.sym.repeat(curprob[j], repeats=beam_size, axis=0) 
                tmp = mx.sym.pick(tmp, prerow[j])
                tmp = mx.sym.reshape(tmp, shape=(-4,1,-1))
                tmpprob.append(tmp)

                # [1, beam_size] -> [beam_size, beam_size]
                # -> [beam_size,]
                tmp = mx.sym.repeat(currow[j], repeats=beam_size, axis=0) 
                tmp = mx.sym.pick(tmp, prerow[j])
                tmprow.append(tmp)

            precol = tmpcol 
            prerow = tmprow
            preprob = tmpprob

        # list of [1, beam_size], which length is target_seq_len*batch_size
        final_path.extend(curcol)  
        final_path.reverse()

        final_prob.extend(curprob)  
        final_prob.reverse()

        # target_seq_len*batch_size*beam_size
        # -> [target_seq_len*batch_size, beam_size]
        # -> [target_seq_len, batch_size, beam_size]
        # -> [batch_size, target_seq_len, beam_size]
        final_path = mx.sym.concat(*final_path, dim=0)
        final_path = mx.sym.reshape(final_path, shape=(-4,target_seq_len,-1, 0))
        final_path = mx.sym.swapaxes(final_path, dim1=0, dim2=1)

        final_prob = mx.sym.concat(*final_prob, dim=0)
        final_prob = mx.sym.reshape(final_prob, shape=(-4,target_seq_len,-1, 0))
        final_prob = mx.sym.swapaxes(final_prob, dim1=0, dim2=1)

        # [batch_size, target_seq_len, beam_size]
        return final_path, final_prob


    ##
    # @brief a->b in dual-learning model, do a beam-search in training
    #
    # @return 
    def beam_decode(self, model, source_encoded, source_encoded_seq_len, 
            target_embed, target_seq_len, beam_size):
        target_vocab_size = self.config.vocab_target_size

        # [batch_size, 1, source_seq_len, num_hidden]
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
        target_prev = mx.sym.zeros(shape=(self.config.batch_size, beam_size))

        target_row_all = []
        target_col_all = []
        target_prob_all = []

        # decode time step
        for t in range(target_seq_len):
            # 001. get previous target embedding
            # [batch_size, beam_size, num_embed]
            target_embed_prev,_,_ = model.embedding_target.encode(target_prev, self.target_lengths, target_seq_len)

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

            # 004. length penalty
            # TODO

            # 005. get top-k prediction
            # -> [batch_size, beam_size, target_vocab_size]
            # -> [batch_size, beam_size*target_vocab_size]
            pred = mx.sym.reshape(pred, shape=(-4, -1, beam_size, 0))
            pred = mx.sym.reshape(pred, shape=(0,-3)) 

            # [batch_size, beam_size]
            topkval,topkpos = mx.sym.topk(pred, k=beam_size, ret_typ="both", axis=-1)
            topk_pos_row = topkpos/target_vocab_size
            topk_pos_col = topkpos%target_vocab_size

            # 005. save result for loss compute
            target_row_all.append(topk_pos_row)
            target_col_all.append(topk_pos_col)
            target_prob_all.append(topkval)

            # 006. update target_prev
            # [batch_size, beam_size] 
            target_prev = topk_pos_col

        # [batch_size, target_seq_len, beam_size]
        path, prob = self._get_path_from_beam(target_row_all, target_col_all, target_prob_all, target_seq_len) 

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

        # target embedding
        # 
        # target_embed: [batch_size, target_seq_len, target_hidden_size] 
        (target_embed,
         target_embed_lengths,
         target_embed_seq_len) = model.embedding_target.encode(self.target, self.target_lengths, target_seq_len)

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
        source_encoded = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1)

        # beam_decoded: [batch_size, target_seq_len, target_vocab_size]
        # beam_path:    [batch_size, target_seq_len, beam_size]
        beam_path, path_prob = self.beam_decode(model, source_encoded, source_encoded_seq_len, 
            target_embed, target_embed_seq_len, self.config.beam_size)

        return beam_path, path_prob


    def backward(self, model, bucket, beam_out):
        '''
        PARAMETERS:
            - beam_path:    [batch_size, target_seq_len, beam_size]
        '''
        target_seq_len, source_seq_len = bucket
        beam_size = self.config.beam_size

        # [batch_size, target_seq_len, beam_size]
        # -> [batch_size, beam_size, target_seq_len]
        # -> [batch_size*beam_size, target_seq_len]
        inputs = mx.sym.swapaxes(beam_out, dim1=1, dim2=2)
        inputs = mx.sym.reshape(inputs, shape=(-3, 0))
        #inputs = mx.sym.BlockGrad(inputs)

        # [batch_size, 1] -> [batch_size, beam_size] -> [batc_size*beam_size,]
        source_lengths = mx.sym.expand_dims(self.target_lengths, axis=1)
        source_lengths = mx.sym.repeat(source_lengths, repeats=beam_size, axis=1)
        source_lengths = mx.sym.reshape(source_lengths, shape=(-1,))

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

        # TODO: is this neccessary?
        #
        # target_decoded: (batch_size*beam_size*source_seq_len, rnn_num_hidden)
        #target_decoded = mx.sym.reshape(data=target_decoded, shape=(-3, 0))

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

        # [batch_size, target_seq_len, beam_size]
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
        # [batch_size, source_seq_len, beam_size] -> [batch_size, beam_size]
        path_logits = -mx.sym.log(path_prob+1e-8)
        path_prob = mx.sym.sum(path_prob, axis=1) 

        # [batch_size, 1]
        prob_sum = mx.sym.sum(path_prob, axis=1, keepdims=True)

        # [batch_size, beam_size] -> [batch_size*beam_size]
        path_prob = mx.sym.broadcast_div(path_prob, prob_sum)  
        path_prob = mx.sym.reshape(path_prob, shape=(-1,))

        # STEP 2. language model score 
        # [batch_size, target_seq_len, beam_size]
        # -> [batch_size, beam_size, target_seq_len]
        # -> [batch_size*beam_size, target_seq_len]
        # -> [batch_size*beam_size, target_seq_len]: -log(p(y|model))
        # -> [batch_size*beam_size]
        forward_path = mx.sym.swapaxes(forward_path, dim1=1, dim2=2) 
        forward_path = mx.sym.reshape(forward_path, shape=(-3, 0)) 
        lm_logits = mx.sym.Custom(data=forward_path, op_type='lm_score', 
                prefix=self.config.lm_prefix, epoch=self.config.lm_epoch, pad=C.PAD_ID,
                devid=self.config.lm_device_ids)
        lm_logits = mx.sym.BlockGrad(lm_logits)
        lm_score = mx.sym.sum(lm_logits, axis=-1)

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
        alpha = self.config.alpha
        #loss = (alpha*lm_score + (1-alpha)*backward_score) * path_prob 
        #loss = (alpha*lm_score + (1-alpha)*backward_score) / self.config.beam_size
        #loss = mx.sym.sum(loss)

        loss = mx.sym.Custom(lm_score=lm_score, path_prob=path_prob, backward_score=backward_score, 
                 op_type='dual_output', alpha=self.config.alpha, scale=self.config.forward_gradient_scale)
        return [loss]
        #return [mx.sym.make_loss(loss), mx.sym.make_loss(lm_score), mx.sym.make_loss(backward_score)]
