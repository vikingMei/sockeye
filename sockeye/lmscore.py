#!/usr/bin/env python3
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

g_model = None


class LMScoreConfig(object):
    def __init__(self, prefix:str, epoch:int, pad:Optional[int] = 0, devid:Optional[int] = 0) -> None:
        '''
        PARAMETERS:
            - prefix: prefix to load lstm model
            - epoch: epoch os model to load
            - bptt: lstm unroll length
            - batch_size: batch size

        TODO:
            1. using bucking lstm, remove bptt 
            2. remove batch size
            3. build from args at start up
        '''
        self.prefix = prefix 
        self.epoch = int(epoch)
        self.pad = int(pad)

        self.data_name = 'data'
        self.label_name = 'label'

        # TODO: read gpu id from config
        devid = int(devid)
        self.context = mx.gpu(devid)



@mx.operator.register("lm_score")
class LMScoreProp(mx.operator.CustomOpProp):

    def __init__(self, prefix, epoch, pad = 0, devid=0):
        super(LMScoreProp, self).__init__(need_top_grad=False)

        config = LMScoreConfig(prefix=prefix, epoch=epoch, pad=pad, devid=devid)
        self.config = config

        # load module
        global g_model
        if not g_model:
            logger.info('load launguage model from [%s-%04d.params], run on device: [%d]', prefix, int(epoch), int(devid))
            g_model = mx.module.Module.load(config.prefix, config.epoch, 
                label_names=[config.label_name], data_names=[config.data_name], context=config.context)
        self.model = g_model

    def create_operator(self, ctx, shapes, dtypes):
        return LMScore(self.model, self.config)

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

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        '''
        don't have to compute backward gradients
        '''
        return []



class LMScore(mx.operator.CustomOp):
    def __init__(self, model, config:LMScoreConfig) -> None:
        '''
        load ngram module from file 
        '''
        super(LMScore, self).__init__()

        self.model = model
        self.config = config


    def forward(self, is_train, req, in_data, out_data, aux):
        '''
        compute ppl from input
        '''
        data = in_data[0]

        label = mx.nd.full(data.shape, self.config.pad)
        seqlen = data.shape[1]
        label[:,0:seqlen-1]  = data[:,1:]

        # build data batch
        provide_data = [mx.io.DataDesc(name=self.config.data_name, shape=(data.shape), layout=C.BATCH_MAJOR)]
        provide_label = [mx.io.DataDesc(name=self.config.label_name, shape=(label.shape), layout=C.BATCH_MAJOR)]

        batch = mx.io.DataBatch([data], [label], pad=self.config.pad, provide_data=provide_data)

        # TODO: test the bind function, whether is necessary every time 
        self.model.bind(data_shapes=provide_data, label_shapes=provide_label, for_training=False, force_rebind=True)
        self.model.forward(batch)

        pred = self.model.get_outputs()[-1]
        pred = pred.reshape(data.shape)

        flag = (label==self.config.pad).as_in_context(pred.context)
        pred = pred*(1-flag)

        self.assign(out_data[0], req[0], pred)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        return
