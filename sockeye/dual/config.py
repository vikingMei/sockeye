#!/usr/bin/env python
# coding: utf-8

from typing import Dict, List, Optional, Tuple
from sockeye.config import Config

class DualConfig(Config):
    def __init__(self, lm_prefix:str, lm_epoch:int, beam_size:int, batch_size:int, alpha:float,
            vocab_source_size:int, vocab_target_size:int):
        super().__init__()

        self.lm_prefix = lm_prefix
        self.lm_epoch = lm_epoch
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.vocab_source_size = vocab_source_size
        self.vocab_target_size = vocab_target_size
