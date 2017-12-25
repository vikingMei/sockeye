#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(auimoviki@gmail.com)

import copy
import mxnet as mx

from .. import constants as C
from .. import model
from .. import utils
from .. import data_io
from .. import loss

from typing import AnyStr, List, Optional

class ModelBuilder():
    def __init__(self, context: List[mx.context.Context],
            config: model.ModelConfig,
            train_iter: data_io.ParallelBucketSentenceIter, logger) -> None:
        self.config = copy.deepcopy(config)
        self.config.freeze()

        self.context = context
        self.logger = logger

        utils.check_condition(train_iter.pad_id == C.PAD_ID == 0, "pad id should be 0")

        self.source = mx.sym.Variable(C.SOURCE_NAME)
        self.source_lengths = utils.compute_lengths(self.source)

        self.target = mx.sym.Variable(C.TARGET_NAME)
        self.target_lengths = utils.compute_lengths(self.target)

        self.labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        self.data_names = [x[0] for x in train_iter.provide_data]
        self.label_names = [x[0] for x in train_iter.provide_label]

        self.default_bucket_key=train_iter.default_bucket_key
        self.max_bucket_key = train_iter.buckets[0]


    def sym_gem_predict(self, seq_lens, prefix=""):
        """
        Returns a (grouped) loss symbol given source & target input lengths.
        Also returns data and label names for the BucketingModule.
        """
        raise NotImplementedError()


    def get_loss(self, logits):
        model_loss = loss.get_loss(self.config.config_loss)
        return model_loss.get_loss(logits, self.labels)
         

    def build(self, bucketing:bool, prefix=""):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        def sym_gen(seq_len):
            logits = self.sym_gen_predict(seq_len, prefix)
            loss = self.get_loss(logits)

            return mx.sym.Group(loss), self.data_names, self.label_names

        if bucketing:
            self.logger.info("Using bucketing. Default max_seq_len=%s", self.default_bucket_key)
            return mx.mod.BucketingModule(sym_gen=sym_gen,
                                          logger=self.logger,
                                          default_bucket_key=self.default_bucket_key,
                                          context=self.context)
        else:
            self.logger.info("No bucketing. Unrolled to (%d,%d)",
                        self.config.max_seq_len_source, self.config.max_seq_len_target)
            symbol, _, __ = sym_gen(self.max_bucket_key)

            return mx.mod.Module(symbol=symbol,
                                 data_names=self.data_names,
                                 label_names=self.label_names,
                                 logger=self.logger,
                                 context=self.context)
