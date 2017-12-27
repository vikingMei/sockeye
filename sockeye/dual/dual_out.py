#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(auimoviki@gmail.com)

import pdb 
import logging
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Union, Set
from . import constants as C

import mxnet as mx

logger = logging.getLogger(__name__)


class DualOutConfig(object):
    def __init__(self):
        pass


@mx.operator.register("lm_score")
class DualOutProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(DualOutProp, self).__init__(need_top_grad=False)

    def create_operator(self, ctx, shapes, dtypes):
        return DualOut(self.model, self.config)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return in_shape, [output_shape], []

    def infer_type(self, in_type):
        return in_type, [in_type[0]], []



class DualOut(mx.operator.CustomOp):
    def __init__(self, model, config:DualOutConfig) -> None:
        '''
        load ngram module from file 
        '''
        super(DualOut, self).__init__()

        self.model = model
        self.config = config


    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        input: 
        compute ppl from input
        '''
        pass


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass
