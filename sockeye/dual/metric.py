#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(auimoviki@gmail.com)

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
                 beam_size: int, 
                 name: str = C.DUAL,
                 output_names: Optional[List[str]] = None,
                 label_names: Optional[List[str]] = None) -> None:
        super().__init__(name, output_names=None, label_names=None)
        self.name = [C.DUAL, 'forward_ppl', 'backward_ppl']
        self.beam_size = beam_size
        self.sum_metric = [0,0,0]


    def update(self, labels, preds):
        """
        the reward have been compute in loss comute 
        """
        reward = preds[1]
        target_label = labels[0]
        batch_size = target_label.shape[0] 

        for i in range(0,3):
            self.sum_metric[i] += preds[i].sum().asscalar()
        self.num_inst += batch_size


    def reset(self):
        self.num_inst = 0
        self.sum_metric = [0.0, 0.0, 0.0]


    def get(self):
        if self.num_inst == 0:
            return (self.name, float('nan'))
        else:
            sum_metric = [0,0,0]
            for i in range(0,3):
                sum_metric[i] = self.sum_metric[i]/self.num_inst
            return (self.name, sum_metric)
