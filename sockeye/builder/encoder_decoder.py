#!/usr/bin/env python
# coding: utf-8
#
# Usage: 
# Author: wxm71(weixing.mei@aispeech.com)

from .. import constants as C
from .. import model
from .. import utils
from .. import data_io
from .. import loss
from .builder import ModelBuilder

import copy
import mxnet as mx
from typing import AnyStr, List, Optional


class EncoderDecoderBuilder(ModelBuilder):
    def _sym_gen_predict(self, seq_lens, prefix=""):
        """
        Returns a (grouped) loss symbol given source & target input lengths.
        Also returns data and label names for the BucketingModule.
        """
        source_seq_len, target_seq_len = seq_lens

        module_factory = model.SockeyeModel(self.config, prefix)
        module_factory._build_model_components()

        # source embedding
        (source_embed,
         source_embed_length,
         source_embed_seq_len) = module_factory.embedding_source.encode(self.source, self.source_lengths, source_seq_len)

        # target embedding
        (target_embed,
         target_embed_length,
         target_embed_seq_len) = module_factory.embedding_target.encode(self.target, self.target_lengths, target_seq_len)

        # encoder
        # source_encoded: (source_encoded_length, batch_size, encoder_depth)
        (source_encoded,
         source_encoded_length,
         source_encoded_seq_len) = module_factory.encoder.encode(source_embed, source_embed_length, source_embed_seq_len)

        # decoder
        # target_decoded: (batch_size, target_len, decoder_depth)
        target_decoded = module_factory.decoder.decode_sequence(source_encoded, source_encoded_length, source_encoded_seq_len,
                                                      target_embed, target_embed_length, target_embed_seq_len)

        # target_decoded: (batch_size * target_seq_len, rnn_num_hidden)
        target_decoded = mx.sym.reshape(data=target_decoded, shape=(-3, 0))

        # output layer
        # logits: (batch_size * target_seq_len, target_vocab_size)
        logits = module_factory.output_layer(target_decoded)

        return logits
