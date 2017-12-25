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

from ..builder import ModelBuilder

import pdb
import copy
import mxnet as mx
from typing import AnyStr, List, Optional

import inspect

def PrintFrame():
  callerframerecord = inspect.stack()[2]    # 0 represents this line
                                            # 1 represents line at caller
  frame = callerframerecord[0]
  info = inspect.getframeinfo(frame)
  return info.lineno


def debug_shape(sym):
    lineno = PrintFrame() 

    if int==type(sym):
        print(lineno, sym)
        return 

    inputs = sym.list_inputs()
    fsrc = 0
    ftgt = 0
    flab = 0

    srcshape=(40,40)
    tgtshape=(40,36)

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
    def __init__(self, context: List[mx.context.Context],
            config: model.ModelConfig,
            train_iter: data_io.ParallelBucketSentenceIter, 
            logger, k) -> None:
        super().__init__(context, config, train_iter, logger)
        self.beam_size = k
        self.prefix = 'dual_'

        # [batch_size, target_seq_len] 
        # -> [batch_size, 1, target_seq_len] 
        # -> [batch_size, beam_size, target_seq_len] 
        # -> [batch_size*beam_size, target_seq_len] 
        # -> [batch_size*beam_size*target_seq_len] 
        labels = mx.sym.Variable(C.TARGET_LABEL_NAME)
        labels = mx.sym.expand_dims(labels, axis=1) 
        labels = mx.sym.repeat(labels, repeats=self.beam_size, axis=1) 
        self.labels = mx.sym.reshape(labels, shape=(-1,))

        #self.label_names = []


    ##
    # @brief get path from beam search result 
    #
    # @param rows, a list whoes length is target_seq_len, contain items in size [batch_size,beam_size]
    # @param cols, same as row
    # @param target_seq_len, the length of rows and cols 
    #
    # @return valid path generate from rows and cols, of size [batch_size, target_seq_len, beam_size]
    def _get_path_from_beam(self, rows, cols, target_seq_len):
        final_path = []

        # [batch_size, beam_size]
        # TODO: read batch_size from config
        batch_size = 20
        beam_size = self.beam_size

        # list of [1,beam_size], of length batch_size
        precol = mx.sym.split(cols[target_seq_len-1], axis=0, num_outputs=batch_size)
        prerow = mx.sym.split(rows[target_seq_len-1], axis=0, num_outputs=batch_size, squeeze_axis=1)

        for i in range(target_seq_len-2, -1, -1):
            final_path.extend(precol)

            # list of [1,beam_size], of length beam_size
            curcol = mx.sym.split(cols[i], axis=0, num_outputs=batch_size)  
            currow = mx.sym.split(rows[i], axis=0, num_outputs=batch_size)

            tmpcol = []
            tmprow = []
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
                tmp = mx.sym.repeat(currow[j], repeats=beam_size, axis=0) 
                tmp = mx.sym.pick(tmp, prerow[j])
                tmprow.append(tmp)

            precol = tmpcol 
            prerow = tmprow

        # list of [1, beam_size], which length is target_seq_len*batch_size
        final_path.extend(curcol)  
        final_path.reverse()

        # target_seq_len*batch_size*beam_size
        # -> [target_seq_len*batch_size, beam_size]
        # -> [target_seq_len, batch_size, beam_size]
        final_path = mx.sym.concat(*final_path, dim=0)
        final_path = mx.sym.reshape(final_path, shape=(-4,target_seq_len,-1, 0))

        # [batch_size, target_seq_len, beam_size]
        return mx.sym.swapaxes(final_path, dim1=0, dim2=1)


    ##
    # @brief a->b in dual-learning model, do a beam-search in training
    #
    # @return 
    def beam_decode(self, model, source_encoded, source_encoded_seq_len, 
            target_embed, target_seq_len, beam_size):
        target_vocab_size = self.config.vocab_target_size

        # list of [batch_size, 1, num_embed], which length is target_seq_len
        target_embed_split = mx.sym.split(target_embed, num_outputs=target_seq_len, axis=1)

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
            pred = mx.sym.reshape(pred, shape=(-4, -1, beam_size, 0))
            pred = mx.sym.reshape(pred, shape=(0,-3)) 

            # [batch_size, beam_size]
            topkval,topkpos = mx.sym.topk(pred, k=beam_size, ret_typ="both", axis=-1)
            topk_pos_row = topkpos/target_vocab_size
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
        beam_decoded, beam_path = self.beam_decode(model, source_encoded, source_encoded_seq_len, 
            target_embed, target_embed_seq_len, self.beam_size)

        return beam_decoded, beam_path


    def backward(self, model, bucket, beam_out):
        '''
        PARAMETERS:
            - beam_path:    [batch_size, target_seq_len, beam_size]
        '''
        target_seq_len, source_seq_len = bucket
        beam_size = self.beam_size

        # [batch_size, target_seq_len, beam_size]
        # -> [batch_size, beam_size, target_seq_len]
        # -> [batch_size*beam_size, target_seq_len]
        inputs = mx.sym.swapaxes(beam_out, dim1=1, dim2=2)
        inputs = mx.sym.reshape(inputs, shape=(-3, 0))
        inputs = mx.sym.BlockGrad(inputs)

        # [batch_size, 1] -> [batch_size, beam_size] -> [batc_size*beam_size,]
        source_lengths = mx.sym.expand_dims(self.target_lengths, axis=1)
        source_lengths = mx.sym.repeat(source_lengths, repeats=beam_size, axis=1)
        source_lengths = mx.sym.reshape(source_lengths, shape=(-1,))

        # [batch_size, 1] -> [batch_size, beam_size] -> [batc_size*beam_size,]
        target_lengths = mx.sym.expand_dims(self.source_lengths, axis=1)
        target_lengths = mx.sym.repeat(target_lengths, repeats=beam_size, axis=1)
        target_lengths = mx.sym.reshape(target_lengths, shape=(-1,))

        # [batch_size, target_seq_len] -> [batch_size, 1, target_seq_len]
        # -> [batch_size, beam_size, target_seq_len] 
        # -> [batc_size*beam_size, target_seq_len]
        target = mx.sym.expand_dims(self.source, axis=1)
        target = mx.sym.repeat(target, repeats=beam_size, axis=1)
        target = mx.sym.reshape(target, shape=(-3, 0))

        # inputs embedding
        # [batch_size*beam_size, source_seq_len, num_embed]
        (source_embed,
         source_embed_length,
         source_embed_seq_len) = model.embedding_source.encode(inputs, source_lengths, source_seq_len)

        # target embedding
        # [batch_size*beam_size, target_seq_len, num_embed]
        (target_embed,
         target_embed_length,
         target_embed_seq_len) = model.embedding_target.encode(target, target_lengths, target_seq_len)

        # encoder
        # source_encoded: (source_encoded_length, batch_size*beam_size, encoder_depth)
        (source_encoded,
         source_encoded_length,
         source_encoded_seq_len) = model.encoder.encode(source_embed, source_embed_length, source_embed_seq_len)

        # decoder
        # target_decoded: (batch_size*beam_size, target_len, decoder_depth)
        target_decoded = model.decoder.decode_sequence(source_encoded, source_encoded_length, source_encoded_seq_len,
                                                      target_embed, target_embed_length, target_embed_seq_len)

        # target_decoded: (batch_size*beam_size*target_seq_len, rnn_num_hidden)
        target_decoded = mx.sym.reshape(data=target_decoded, shape=(-3, 0))

        # output layer
        # logits: (batch_size*beam_size*target_seq_len, target_vocab_size)
        pred = model.output_layer(target_decoded)

        return pred


    def sym_gen_predict(self, seq_lens, prefix=""):
        """
        Returns a (grouped) loss symbol given source & target input lengths.
        Also returns data and label names for the BucketingModule.
        """
        model_forward = model.SockeyeModel(self.config, "%sf_"%self.prefix)
        model_forward._build_model_components()

        model_backward = model.SockeyeModel(self.config, "%sb_"%self.prefix)
        model_backward._build_model_components()

        # forward_decoded: [batch_size, target_seq_len, target_vocab_size]
        # forward_path:    [batch_size, target_seq_len, beam_size]
        forward_decoded, forward_path = self.forward(model_forward, seq_lens)

        # [batch_size*beam_size*target_seq_len, target_vocab_size]
        backward_decoded = self.backward(model_backward, seq_lens, forward_path)

        return [forward_path,backward_decoded]

    def get_loss(self, logits):
        forward_path,backward_logits = logits

        # [batch_size, target_seq_len, beam_size]
        # -> [batch_size, beam_size, target_seq_len]
        # -> [batch_size*beam_size, target_seq_len]
        forward_path = mx.sym.swapaxes(forward_path, dim1=1, dim2=2) 
        forward_path = mx.sym.reshape(forward_path, shape=(-3, 0)) 

        # [batch_size*beam_size, target_seq_len]: -log(p(y|model))
        pad = 0
        epoch = 39
        prefix = './lm/model'

        forward_logits = mx.sym.Custom(forward_path, name='lm_score', op_type='lm_score',
                prefix=prefix, epoch=epoch, pad=pad)

        # backward loss
        model_loss = loss.get_loss(self.config.config_loss)

        return model_loss.get_loss(backward_logits, self.labels, self.source, self.beam_size, forward_logits)

