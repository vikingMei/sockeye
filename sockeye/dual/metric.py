#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import pdb
import logging
import mxnet as mx

from typing import List, Optional
from mxnet.metric import EvalMetric

from .. import constants as C
from ..loss import LossConfig

logger = logging.getLogger(__name__)



class DualMetric(EvalMetric):
    """
    Version of the cross entropy metric that ignores padding tokens.

    :param loss_config: The configuration used for the corresponding loss.
    :param name: Name of this metric instance for display.
    :param output_names: Name of predictions that should be used when updating with update_dict.
    :param label_names: Name of labels that should be used when updating with update_dict.
    """

    def __init__(self,
                 name: str = C.DUAL,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None) -> None:
        super().__init__(name, output_names=None, label_names=None)

        # TODO: read from config
        self.alpha = 0.5

    def update(self, labels, preds):
        pdb.set_trace()

        # TODO: 
        #
        #   1. see dual/loss.py, get_loss function
        #   2. add label smoothing and normalization from loss.py
        #   3. check the size of labels and preds, in Perplexity, there is a zip loop
        #
        # fpred:  [batch_size*beam_size, target_seq_len] -log(p(y_{i+1}|y_i)
        # bpred:  [batch_size*beam_size*target_seq_len, source_vocab_size], 
        #                  softmax output of distribution on source vocabulary
        # label:  [batch_size*beam_size*target_seq_len]
        fpred, bpred, _, label = preds

        batch_size = fpred.shape[0]

        floss = self.alpha*mx.nd.sum(fpred)

        # [batch_size*beam_size*target_seq_len]
        bprob = mx.nd.pick(bpred, label.astype(dtype="int32"))
        ignore = (label == C.PAD_ID).astype(dtype=bpred.dtype)

        bprob = bprob * (1 - ignore) + ignore
        bloss = -mx.nd.log(bprob + 1e-8)    # pylint: disable=invalid-unary-operand-type
        bloss = (1-self.alpha) * mx.nd.sum(bloss)

        self.sum_metric += (floss+bloss).asscalar()
        self.num_inst += batch_size

        return loss
