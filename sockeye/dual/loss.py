#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(auimoviki@gmail.com)


import logging
import mxnet as mx
from typing import List, Optional

from .. import constants as C
from ..loss import Loss, LossConfig

logger = logging.getLogger(__name__)


class DualLoss(Loss):
    """
    Computes the dual-learning loss.

    :param loss_config: Loss configuration.
    """
    def __init__(self, loss_config: LossConfig) -> None:
        logger.info("Loss: Dual(normalization_type=%s, label_smoothing=%s)",
                    loss_config.normalization_type, loss_config.label_smoothing)
        self.loss_config = loss_config

    def get_loss(self, forward_logits, path_prob, backward_logits, labels, source, beam_size) -> List[mx.sym.Symbol]:
        """
        Returns loss and softmax output symbols given logits and integer-coded labels.

        :param logits: Shape: (batch_size * target_seq_len, target_vocab_size).
        :param labels: Shape: (batch_size * target_seq_len,).
        :return: List of loss symbol.
        """
        if self.loss_config.normalization_type == C.LOSS_NORM_VALID:
            normalization = "valid"
        elif self.loss_config.normalization_type == C.LOSS_NORM_BATCH:
            normalization = "null"
        else:
            raise ValueError("Unknown loss normalization type: %s" % self.loss_config.normalization_type)

        # [batch_size, target_seq_len] 
        # -> [batch_size, 1, target_seq_len] 
        # -> [batch_size, beam_size, target_seq_len] 
        # -> [batch_size*beam_size, target_seq_len] 
        # -> [batch_size*beam_size*target_seq_len] 
        sources = mx.sym.expand_dims(sources, axis=1, name="loss_expand_source") 
        sources = mx.sym.repeat(sources, repeats=beam_size, axis=1, name='loss_source_repeat1') 
        sources = mx.sym.reshape(sources, shape=(-3,0), name='loss_source_reshape1')
        sources = mx.sym.reshape(sources, shape=(-3,), name='loss_source_reshape2')

        forward_logits = mx.sym.BlockGrad(forward_logits)

        # TODO: merge the last two into dataIter.label
        #
        # the last two is a temporary solution
        #   1. make label appear in final compute graph, stop runtime complain
        #   2. using as label for B->A translation
        return [mx.sym.make_loss(forward_logits, name='lm_score'),
                mx.sym.SoftmaxOutput(data=backward_logits,
                                     label=sources,
                                     ignore_label=C.PAD_ID,
                                     use_ignore=True,
                                     normalization=normalization,
                                     smooth_alpha=self.loss_config.label_smoothing,
                                     name='reduction_score'),
                mx.sym.make_loss(labels, name=C.TARGET_LABEL_NAME), 
                mx.sym.make_loss(sources, name='makeloss_on_source')]

    def create_metric(self) -> "DualMetric":
        raise NotImplementedError()
