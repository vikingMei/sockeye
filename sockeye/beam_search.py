#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: weixing.mei(auimoviki@gmail.com)

import pdb 
import logging
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Set
from . import constants as C

import mxnet as mx
import numpy as np

logger = logging.getLogger(__name__)

g_model = None


@mx.operator.register("beam_search")
class BeamSearchProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(BeamSearchProp, self).__init__(need_top_grad=True)

    def create_operator(self, ctx, shapes, dtypes):
        return BeamSearch()

    def list_arguments(self):
        return ['rows', 'cols', 'probs']

    def list_outputs(self):
        return ['path', 'path_prob']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return in_shape, [output_shape, output_shape], []

    def infer_type(self, in_type):
        return in_type, [in_type[0], in_type[0]], []

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        deps = []
        deps.append(out_data[0])
        deps.append(out_grad[1])
        return deps


class BeamSearch(mx.operator.CustomOp):
    def __init__(self) -> None:
        super(BeamSearch, self).__init__()
        # TODO: compute from vocab
        self.eos_id = 3
        self.pad_id = 0


    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        compute ppl from input
        '''

        # [target_seq_len, batch_size, beam_size]
        rows = in_data[0].astype('int32').asnumpy()
        cols = in_data[1].astype('int32').asnumpy()
        probs= in_data[2].asnumpy()
        self.newrow = np.zeros_like(rows, dtype='int32') 

        shape = rows.shape
        seqlen = shape[0]
        batch_size = shape[1] 
        beam_size = shape[2]

        # [1, beam_size]: [[0, 1, 2, 3 ...]]
        # -> [batch_size, beam_size]
        rowidx = np.arange(beam_size)
        rowidx = rowidx.reshape((1,-1))
        rowidx = rowidx.repeat(repeats=batch_size, axis=0)

        final_path = cols.copy()
        final_prob = probs.copy()
        for i in range(seqlen-1, -1, -1):
            for j in range(0, batch_size):
                for k in range(0, beam_size):
                    idx = rowidx[j,k]
                    final_path[i,j,k] = cols[i,j,idx]
                    final_prob[i,j,k] = probs[i,j,idx]

                    self.newrow[i,j,k] = idx
                    rowidx[j,k] = rows[i,j,idx]

        #for i in range(seqlen-2, -1, -1): 
        #    final_path[i+1, :, :] = precol
        #    final_prob[i+1, :, :] = preprob
        #    self.newrow[i+1, :, :] = prerow
        #    for j in range(0, batch_size):
        #        # [beam_size]
        #        # -> [1, beam_size]
        #        # -> [beam_size, beam_size]
        #        # -> [beam_size,]
        #        tmp = cols[i, j, :]
        #        tmp = tmp.expand_dims(axis=0)
        #        tmp = tmp.repeat(repeats=beam_size, axis=0)
        #        tmp = tmp.pick(prerow[j,:], axis=1)
        #        precol[j,:] = tmp
        #        tmp = probs[i, j, :]
        #        tmp = tmp.expand_dims(axis=0)
        #        tmp = tmp.repeat(repeats=beam_size, axis=0)
        #        tmp = tmp.pick(prerow[j,:], axis=1)
        #        preprob[j,:] = tmp
        #        tmp = rows[i, j, :]
        #        tmp = tmp.expand_dims(axis=0)
        #        tmp = tmp.repeat(repeats=beam_size, axis=0)
        #        tmp = tmp.pick(prerow[j,:], axis=1)
        #        prerow[j,:] = tmp
        #
        #final_path[0,:,:] = precol
        #final_prob[0,:,:] = preprob
        #self.newrow[0,:,:] = prerow

        # update eos(as dual_forward is input of dual_backward, 
        # so we don't need to keep eos, just replace it with padding)
        for i in range(0, batch_size):
            for j in range(0, beam_size):
                for t in range(0, seqlen): 
                    if self.eos_id==final_path[t, i, j]:
                        final_path[t:, i, j] = self.pad_id
                        final_prob[t:, i, j] = 0.0
                        break

        self.assign(out_data[0], req[0], final_path)
        self.assign(out_data[1], req[1], final_prob)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        path = out_data[0]
        inbuf = out_grad[1]

        shape = path.shape
        seqlen = shape[0]
        batch_size = shape[1] 
        beam_size = shape[2]

        grad = np.zeros(path.shape)

        ctx = path.context
        for t in range(0, seqlen): 
            for i in range(0, batch_size): 
                for j in range(0, beam_size):
                    if self.pad_id==path[t, i, j]:
                        continue 

                    idx = self.newrow[t, i, j]
                    grad[t, i, idx] += inbuf[t, i, j].asscalar()

        zeros = mx.nd.zeros_like(path)
        self.assign(in_grad[0], req[0], zeros)
        self.assign(in_grad[1], req[1], zeros)
        self.assign(in_grad[2], req[2], mx.nd.array(grad, ctx=ctx))
