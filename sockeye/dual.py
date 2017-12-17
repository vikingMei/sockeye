#!/usr/bin/env python3
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

import pdb
import logging
import mxnet as mx

from . import data_io
from . import training
from . import utils
from . import constants as C
from . import loss
from . import model
from . import dual_model

logger = logging.getLogger(__name__)

class DualModel(training.TrainingModel):
    """
    dual learning translate model
    """

    def _build_module(self, train_iter: data_io.ParallelBucketSentenceIter):
        """
        Initializes model components, creates training symbol and module, and binds it.
        """
        utils.check_condition(train_iter.pad_id == C.PAD_ID == 0, "pad id should be 0")

        # get the length of each input utterance
        #
        # source: [batch_size, source_seq_len]
        # source_length: [batch_size, ]
        source = mx.sym.Variable(C.SOURCE_NAME)
        source_length = utils.compute_lengths(source)

        # get the length of each label utterance 
        #
        # target: [batch_size, target_seq_len]
        # target_length: [batch_size, 1]
        target = mx.sym.Variable(C.TARGET_NAME)
        target_length = utils.compute_lengths(target)

        # target label
        #
        # labels: [batch_size, target_seq_len]
        labels = mx.sym.reshape(data=mx.sym.Variable(C.TARGET_LABEL_NAME), shape=(-1,))

        # get loss function
        model_loss = loss.get_loss(self.config.config_loss)

        data_names = [x[0] for x in train_iter.provide_data]
        label_names = [x[0] for x in train_iter.provide_label]

        def sym_gen(seq_lens):
            """
            Returns a (grouped) loss symbol given source & target input lengths.
            Also returns data and label names for the BucketingModule.
            """
            BEAM_SIZE = 2
            BATCH_SIZE=40

            # source_seq_len: max input utterance length
            # target_seq_len: max output utterance length
            source_seq_len, target_seq_len = seq_lens

            # source embedding
            #
            # source_embed_length: same as source_length
            # source_embed_seq_len: same as source_seq_len
            # source_embed: [batch_size, source_seq_len, source_hidden_size]
            (source_embed,
             source_embed_length,
             source_embed_seq_len) = self.embedding_source.encode(source, source_length, source_seq_len)

            # target embedding
            # 
            # target_embed: [batch_size, target_seq_len, target_hidden_size] 
            (target_embed,
             target_embed_lengths,
             target_embed_seq_len) = self.embedding_target.encode(target, target_length, target_seq_len)


            # encoder
            #
            # source_encoded_length: [batch_size, ], same as source_embed_length
            # source_encoded_seq_len: same as source_embed_seq_len, max length of input utterances
            # source_encoded: [source_seq_len, batch_size, encoded_hidden_len]
            (source_encoded,
             source_encoded_lengths,
             source_encoded_seq_len) = self.encoder.encode(source_embed,
                                                           source_embed_length,
                                                           source_embed_seq_len)
            source_encoded = mx.sym.swapaxes(source_encoded, dim1=0, dim2=1)

            # TODO: put BEAM_SIZE to config
            BEAM_SIZE = 3
            # beam_decoded: [batch_size, target_seq_len, target_vocab_size]
            # beam_path:    [batch_size, target_seq_len, beam_size]
            beam_decoded, beam_path = dual_model.beam_decode_model(self, BEAM_SIZE,
                source_encoded, source_encoded_lengths, source_encoded_seq_len, 
                target, target_embed, target_embed_lengths, target_embed_seq_len)

            # do B->A translate
            # -> [batch_size, beam_size, target_seq_len]
            # -> [batch_size*beam_size, target_seq_len]
            #beam_path = mx.sym.swapaxes(beam_path, dim1=1, dim2=2)
            #beam_path = mx.sym.reshape(beam_path, shape=(-3, 0))

            # compute loss
            # [batch_size*target_seq_len, target_vocab_size]
            beam_decoded = mx.sym.reshape(beam_decoded, shape=(-3, 0))

            # label: batch_size*target_seq_len
            probs = model_loss.get_loss(beam_decoded, labels)

            return mx.sym.Group(probs), data_names, label_names

        if self.bucketing:
            logger.info("Using bucketing. Default max_seq_len=%s", train_iter.default_bucket_key)
            return mx.mod.BucketingModule(sym_gen=sym_gen,
                                          logger=logger,
                                          default_bucket_key=train_iter.default_bucket_key,
                                          context=self.context)
        else:
            logger.info("No bucketing. Unrolled to (%d,%d)",
                        self.config.max_seq_len_source, self.config.max_seq_len_target)
            symbol, _, __ = sym_gen(train_iter.buckets[0])
            return mx.mod.Module(symbol=symbol,
                                 data_names=data_names,
                                 label_names=label_names,
                                 logger=logger,
                                 context=self.context)

