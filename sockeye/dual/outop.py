#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(auimoviki@gmail.com)

import pdb 
import logging
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Set

import mxnet as mx
import numpy as np
from .. import constants as C

logger = logging.getLogger(__name__)


class DualOutConfig(object):
    def __init__(self):
        pass


@mx.operator.register("dual_output")
class DualOutProp(mx.operator.CustomOpProp):
    def __init__(self, alpha, scale, beam_size):
        super(DualOutProp, self).__init__(need_top_grad=False)
        self.alpha = float(alpha)
        self.scale = float(scale)
        self.beam_size = int(beam_size)

    def create_operator(self, ctx, shapes, dtypes):
        return DualOut(self.alpha, self.scale, self.beam_size)

    def list_arguments(self):
        return ['lm_score', 'beam_path', 'path_logits', 'backward_pred', 'label']

    def list_outputs(self):
        return ['output', 'lm_score', 'backward_score', 'backward_prob']

    def infer_shape(self, in_shape):
        lm_shape = in_shape[0]
        back_shape = in_shape[3]
        out_shape = [lm_shape[0]]
        return in_shape, [out_shape, out_shape, out_shape, back_shape], []

    def infer_type(self, in_type):
        typ = in_type[0]
        return in_type, [typ,typ,typ,typ], []

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        deps = []
        deps.append(in_data[1])     # beam_path
        deps.append(in_data[4])     # label
        deps.append(out_data[0])    # loss
        deps.append(out_data[3])    # back_prob
        return deps



class DualOut(mx.operator.CustomOp):
    def __init__(self, alpha, scale, beam_size) -> None:
        '''
        load ngram module from file 
        '''
        super(DualOut, self).__init__()
        self.alpha = alpha
        self.scale = scale
        self.uni_prob = 1.0/beam_size
        self.beam_size = beam_size


    def forward(self, is_train, req, in_data, out_data, aux):
        # name notation
        # - pred: output of model, 
        # - prob: probability, name, prob = softmax(pred) 
        # - logits: log(prob)
        # - score: loss value, score = loss(prob)

        # LM(s_mid) [batch_size*beam_size, target_seq_len]
        lm_score = in_data[0]

        # lab_AB    [batch_size*beam_size, target_seq_len]
        beam_path = in_data[1].astype('int32')

        # P(s_mid|s, AB)    [batch_size*beam_size, target_seq_len]
        # probability for each path unit
        path_logits = in_data[2]

        # P(s|s_mid, BA)    [batch_size*beam_size, source_seq_len, source_vocab_size]
        back_pred = in_data[3]

        # [batch_size*beam_size, source_seq_len]
        label = in_data[4] 

        # STEP 1. compute P(s_mid|s) 
        #
        # [batch_size*beam_size, target_seq_len] 
        # -> [batch_size*beam_size, 1]
        valid = C.PAD_ID!=beam_path
        validlen = valid.sum(axis=-1).astype('float32')
        lm_score = lm_score/validlen

        # STEP 2. compute logP(s|s_mid)
        #
        # [batch_size*beam_size, source_seq_len, source_vocab_size]
        # -> [batch_size*beam_size, source_seq_len]
        # -> [batch_size*beam_size, source_seq_len]
        # -> [batch_size*beam_size, source_seq_len]
        # -> [batch_size*beam_size,]
        valid = C.PAD_ID!=label 
        back_prob = back_pred.softmax()
        back_tgt = back_prob.pick(label)
        back_logits = -mx.nd.log(back_tgt+1e-8)
        back_logits = valid*back_logits
        back_score  = back_logits.sum(axis=-1)
        valid = label!=C.PAD_ID
        back_score /= valid.sum(axis=-1).astype('float32')

        # for simplify, just remove the path_prob
        #loss = (self.alpha*lm_score + (1-self.alpha)*back_score)*allpath_prob
        loss = (self.alpha*lm_score + (1-self.alpha)*back_score)

        self.assign(out_data[0], req[0], loss)  
        self.assign(out_data[1], req[1], lm_score)
        self.assign(out_data[2], req[2], back_score)
        self.assign(out_data[3], req[3], back_prob)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        # [batch_size*beam_size]
        loss = out_data[0]

        # lab_AB            [batch_size*beam_size, target_seq_len]
        beam_path = in_data[1]

        # P(s|s_mid, BA)    [batch_size*beam_size, source_seq_len, source_vocab_size]
        back_prob = out_data[3]

        label = in_data[4] 

        # dloss/logP(s_mid|s,AB) 
        # [batch_size*beam_size]
        # -> [batch_size, beam_size]
        # -> [batch_size, 1]
        # -> [batch_size, beam_size]
        # -> [batch_size*beam_size, 1]
        loss = loss.reshape(shape=(-1, self.beam_size))
        loss_mean = loss.mean(axis=-1, keepdims=True)
        reward = loss_mean-loss
        reward = reward.reshape(shape=(-1,1))

        valid = C.PAD_ID!=beam_path
        grad = mx.nd.broadcast_mul(reward, valid) 
        grad *= self.uni_prob*self.scale
        self.assign(in_grad[2], req[2], grad)

        # dloss/theta_BA
        shape = back_prob.shape
        grad = np.zeros(shape)
        ctx = back_prob.context
        back_prob = back_prob.asnumpy()
        label = label.astype(dtype='int32').asnumpy()
        for i in range(0,shape[0]):
            for j in range(0, shape[1]):
                tgt = label[i,j]
                if C.PAD_ID==tgt:
                    continue
                grad[i,j,tgt] = back_prob[i,j,tgt]-1
        grad *= (1-self.alpha)
        grad *= self.uni_prob
        grad *= 0
        self.assign(in_grad[3], req[3], mx.nd.array(grad, ctx=ctx))
