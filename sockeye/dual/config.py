#!/usr/bin/env python
# coding: utf-8

from typing import Dict, List, Optional, Tuple
from sockeye.config import Config

class DualConfig(Config):
    def __init__(self, lm_prefix:str, lm_epoch:int, lm_device_ids: int, 
            beam_size:int, batch_size:int, 
            alpha:float,
            forward_gradient_scale:float,
            vocab_source_size:int, vocab_target_size:int,
            forward_param:str, backward_param:str):
        super().__init__()

        self.lm_prefix = lm_prefix
        self.lm_epoch = lm_epoch
        self.lm_device_ids = int(lm_device_ids)

        self.beam_size = beam_size
        self.batch_size = batch_size

        self.alpha = alpha
        self.forward_gradient_scale = forward_gradient_scale

        self.vocab_source_size = vocab_source_size
        self.vocab_target_size = vocab_target_size

        self.forward_param = forward_param 
        self.backward_param = backward_param 
