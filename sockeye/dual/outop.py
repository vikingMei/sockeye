#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(auimoviki@gmail.com)

import pdb 
import logging
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Set

import mxnet as mx

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
        """
        forward_path: [batch_size, source_seq_len, beam_size]
        path_logits: [batch_size, beam_size] 
        backward_score: [batch_size*bam_size, source_seq_len, source_vocab_size]  
        """
        return ['lm_score', 'path_prob', 'backward_score', 'target']

    def list_outputs(self):
        return ['output', 'lm_score', 'backward_score']

    def infer_shape(self, in_shape):
        lm_shape = in_shape[0]
        path_shape = in_shape[1]
        back_shape = in_shape[2]
        loss_shape = path_shape
        return in_shape, [loss_shape, lm_shape, back_shape], []


    def infer_type(self, in_type):
        typ = in_type[0]
        return in_type, [typ, typ, typ], []



class DualOut(mx.operator.CustomOp):
    def __init__(self, alpha, scale, beam_size) -> None:
        '''
        load ngram module from file 
        '''
        super(DualOut, self).__init__()
        self.alpha = alpha
        self.scale = scale
        self.path_prob = 1.0/beam_size


    def forward(self, is_train, req, in_data, out_data, aux):
        # [batch_size*beam_size]
        lm_score = in_data[0]
        #path_prob = in_data[1]
        path_prob = self.path_prob
        backward_score = in_data[2]

        loss = (self.alpha*lm_score + (1-self.alpha)*backward_score) * path_prob 

        self.assign(out_data[0], req[0], loss)  
        self.assign(out_data[1], req[1], lm_score)
        self.assign(out_data[2], req[2], backward_score)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        lm_score = in_data[0]
        #path_prob = in_data[1]
        path_prob = self.path_prob
        backward_score = in_data[2]

        self.assign(in_grad[0], req[0], mx.nd.zeros_like(in_data[0]))
        self.assign(in_grad[1], req[1], self.scale*(self.alpha*lm_score + (1-self.alpha)*backward_score))
        self.assign(in_grad[2], req[2], (1-self.alpha)*path_prob)
        self.assign(in_grad[3], req[3], mx.nd.zeros_like(in_data[3]))
