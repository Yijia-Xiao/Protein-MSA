# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer based language model."""

import torch
import torch.nn.functional as F

from megatron import get_args, print_rank_0
from megatron import mpu
from .module import MegatronModule
from megatron.model.transformer import ParallelTransformer
from megatron.model.utils import get_linear_layer
from megatron.model.utils import init_method_normal, scaled_init_method_normal

def parallel_lm_logits(input_, word_embeddings_weight, parallel_output,
                       bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_tensor_model_parallel_region(input_)
    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)
    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_tensor_model_parallel_region(logits_parallel)


def get_language_model(attention_mask_func, num_tokentypes, add_pooler,
                       init_method=None, scaled_init_method=None):
    """Build language model and return along with the key to save."""
    args = get_args()

    if init_method is None:
        init_method = init_method_normal(args.init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)

    # Language model.
    args = [attention_mask_func, init_method, scaled_init_method]
    kwargs = {}
    cls = None
    if mpu.is_pipeline_first_stage() and mpu.is_pipeline_last_stage():
        cls = TransformerLanguageModel
        kwargs['num_tokentypes'] = num_tokentypes
        kwargs['add_pooler'] = add_pooler
    elif mpu.is_pipeline_first_stage() and not mpu.is_pipeline_last_stage():
        cls = TransformerLanguageModelFirstStage
        kwargs['num_tokentypes'] = num_tokentypes
    elif not mpu.is_pipeline_first_stage() and mpu.is_pipeline_last_stage():
        cls = TransformerLanguageModelLastStage
        kwargs['add_pooler'] = add_pooler
    else:
        cls = TransformerLanguageModelIntermediateStage

    # Language model.
    language_model = cls(*args, **kwargs)
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


class Pooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method):
        super(Pooler, self).__init__()
        self.dense = get_linear_layer(hidden_size, hidden_size, init_method)

    def forward(self, hidden_states, sequence_index=0):
        # hidden_states: [b, s, h]
        # sequence_index: index of the token to pool.
        pooled = hidden_states[:, sequence_index, :]
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        return pooled


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 init_method,
                 num_tokentypes=0):
        super(Embedding, self).__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes

        args = get_args()
        self.add_msa_positional_embedding = args.add_msa_positional_embedding
        self.add_post_embedding_layernorm = args.add_post_embedding_layernorm

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, self.hidden_size,
            init_method=self.init_method)
        self._word_embeddings_key = 'word_embeddings'

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(
            max_sequence_length, self.hidden_size)
        self._position_embeddings_key = 'position_embeddings'
        # Initialize the position embeddings.
        self.init_method(self.position_embeddings.weight)

        # MSA positional embedding

        if self.add_msa_positional_embedding:
            # self.msa_max_aligns = get_args().max_aligns
            self.msa_max_aligns = 1024
            self.msa_positional_embedding = torch.nn.Embedding(self.msa_max_aligns,
                                                            self.hidden_size)
            self._msa_positional_embedding_key = 'msa_positional_embeddings'
            # Initialize the msa positoinal embeddings.
            self.init_method(self.msa_positional_embedding.weight)
        else:
            print_rank_0('No MSA positional embedding is added')
        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes,
                                                           self.hidden_size)
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # TODO: add layernorm
        if self.add_post_embedding_layernorm:
            from megatron.model import import_layernorm
            LayerNorm = import_layernorm(args.fp32_residual_connection)
            self.emb_layer_norm_before = LayerNorm(
                    args.hidden_size,
                    eps=args.layernorm_epsilon)
            self._emb_layer_norm_before_key = 'emb_layer_norm_before'
        else:
            print_rank_0('No post embedding layernorm is applied')

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes),
                  flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes,
                                                       self.hidden_size)
        # Initialize the token-type embeddings.
        args = get_args()
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        if self.add_msa_positional_embedding:
            msa_position_ids = torch.arange(input_ids.size(0), device=position_embeddings.device)\
                .reshape(-1, 1)\
                .repeat(1, position_ids.size(1))
            msa_positional_embeddings = self.msa_positional_embedding(msa_position_ids)

            embeddings = words_embeddings + position_embeddings + msa_positional_embeddings
        else:
            embeddings = words_embeddings + position_embeddings

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        if self.add_post_embedding_layernorm:
            embeddings = self.emb_layer_norm_before(embeddings)
        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._word_embeddings_key] \
            = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        state_dict_[self._position_embeddings_key] \
            = self.position_embeddings.state_dict(
                destination, prefix, keep_vars)
        if self.add_msa_positional_embedding:
            state_dict_[self._msa_positional_embedding_key] \
                = self.msa_positional_embedding.state_dict(
                    destination, prefix, keep_vars)
        if self.add_post_embedding_layernorm:
            state_dict_[self._emb_layer_norm_before_key] \
                = self.emb_layer_norm_before.state_dict(
                    destination, prefix, keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] \
                = self.tokentype_embeddings.state_dict(
                    destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] \
                        = state_dict[key]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self._position_embeddings_key in state_dict:
            state_dict_ = state_dict[self._position_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'position_embeddings' in key:
                    state_dict_[key.split('position_embeddings.')[1]] \
                        = state_dict[key]
        self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        if self.add_msa_positional_embedding:
            if self._msa_positional_embedding_key in state_dict:
                state_dict_ = state_dict[self._msa_positional_embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'msa_positional_embeddings' in key:
                        state_dict_[key.split('msa_positional_embeddings.')[1]] \
                            = state_dict[key]
            self.msa_positional_embedding.load_state_dict(state_dict_, strict=strict)

        if self.add_post_embedding_layernorm:
            if self._emb_layer_norm_before_key in state_dict:
                state_dict_ = state_dict[self._emb_layer_norm_before_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if 'emb_layer_norm_before_key' in key:
                        state_dict_[key.split('emb_layer_norm_before_key.')[1]] \
                            = state_dict[key]
            self.emb_layer_norm_before.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tokentype_embeddings' in key:
                        state_dict_[key.split('tokentype_embeddings.')[1]] \
                            = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_,
                                                          strict=strict)
            else:
                print('***WARNING*** expected tokentype embeddings in the '
                      'checkpoint but could not find it', flush=True)


class TransformerLanguageModelBase(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
          masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 attention_mask_func,
                 init_method,
                 output_layer_init_method,
                 num_tokentypes=0,
                 add_pooler=False):
        super(TransformerLanguageModelBase, self).__init__()
        args = get_args()

        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.add_pooler = add_pooler

        # Embeddings.
        if mpu.is_pipeline_first_stage():
            self.embedding = Embedding(self.hidden_size,
                                       args.padded_vocab_size,
                                       args.max_position_embeddings,
                                       args.hidden_dropout,
                                       self.init_method,
                                       self.num_tokentypes)
            self._embedding_key = 'embedding'

        # Transformer.
        self.transformer = ParallelTransformer(
            attention_mask_func, self.init_method, 
            output_layer_init_method)
        self._transformer_key = 'transformer'

        # Pooler.
        if mpu.is_pipeline_last_stage() and self.add_pooler:
            self.pooler = Pooler(self.hidden_size, self.init_method)
            self._pooler_key = 'pooler'

    def forward(self, language_model_input, attention_mask,
                tokentype_ids=None, layer_past=None, get_key_value=False,
                pooling_sequence_index=0):

        # Embeddings.
        if mpu.is_pipeline_first_stage():
            (input_ids, position_ids) = language_model_input
            embedding_output = self.embedding(input_ids, position_ids,
                                              tokentype_ids=tokentype_ids)
            transformer_input = embedding_output
        else:
            transformer_input = language_model_input

        # Transformer.
        transformer_output = self.transformer(transformer_input,
                                              attention_mask,
                                              layer_past=layer_past,
                                              get_key_value=get_key_value)

        if mpu.is_pipeline_last_stage() and self.add_pooler:
            pooled_output = self.pooler(transformer_output,
                                        pooling_sequence_index)
            return transformer_output, pooled_output

        return transformer_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        if mpu.is_pipeline_first_stage():
            state_dict_[self._embedding_key] \
                = self.embedding.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)
        state_dict_[self._transformer_key] \
            = self.transformer.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        if mpu.is_pipeline_last_stage() and self.add_pooler:
            state_dict_[self._pooler_key] \
                = self.pooler.state_dict_for_save_checkpoint(
                    destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if mpu.is_pipeline_first_stage():
            if self._embedding_key in state_dict:
                state_dict_ = state_dict[self._embedding_key]
            else:
                # for backward compatibility.
                state_dict_ = {}
                for key in state_dict.keys():
                    if '_embeddings' in key:
                        state_dict_[key] = state_dict[key]
            self.embedding.load_state_dict(state_dict_, strict=strict)

        # Transformer.
        if self._transformer_key in state_dict:
            state_dict_ = state_dict[self._transformer_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'transformer.' in key:
                    state_dict_[key.split('transformer.')[1]] = state_dict[key]
        self.transformer.load_state_dict(state_dict_, strict=strict)

        # Pooler.
        if mpu.is_pipeline_last_stage() and self.add_pooler:
            #assert 'pooler' in state_dict, \
            #    'could not find data for pooler in the checkpoint'
            if 'pooler' in state_dict:
                self.pooler.load_state_dict(state_dict[self._pooler_key],
                                            strict=strict)


class TransformerLanguageModel(TransformerLanguageModelBase):
    """Transformer language model (see TransformerLanguageModelBase
       for description of arguments).
    """

    def __init__(self,
                 attention_mask_func,
                 init_method,
                 output_layer_init_method,
                 num_tokentypes=0,
                 add_pooler=False):
        super(TransformerLanguageModel, self).__init__(
            attention_mask_func,
            init_method,
            output_layer_init_method,
            num_tokentypes=num_tokentypes,
            add_pooler=add_pooler)

    def forward(self, input_ids, position_ids, attention_mask,
                tokentype_ids=None, layer_past=None, get_key_value=False,
                pooling_sequence_index=0):
        return super(TransformerLanguageModel, self).forward(
            (input_ids, position_ids),
            attention_mask,
            tokentype_ids=tokentype_ids,
            layer_past=layer_past,
            get_key_value=get_key_value,
            pooling_sequence_index=pooling_sequence_index
        )


class TransformerLanguageModelFirstStage(TransformerLanguageModelBase):
    """Transformer language model, first stage (see
       TransformerLanguageModelBase for description of arguments).
    """

    def __init__(self,
                 attention_mask_func,
                 init_method,
                 output_layer_init_method,
                 num_tokentypes=0):
        super(TransformerLanguageModelFirstStage, self).__init__(
            attention_mask_func,
            init_method,
            output_layer_init_method,
            num_tokentypes=num_tokentypes)

    def forward(self, input_ids, position_ids, attention_mask,
                tokentype_ids=None, layer_past=None, get_key_value=False):
        return super(TransformerLanguageModelFirstStage, self).forward(
            (input_ids, position_ids),
            attention_mask,
            tokentype_ids=tokentype_ids,
            layer_past=layer_past,
            get_key_value=get_key_value
        )


class TransformerLanguageModelIntermediateStage(TransformerLanguageModelBase):
    """Transformer language model, intermediate stage (see
       TransformerLanguageModelBase for description of arguments).
    """

    def __init__(self,
                 attention_mask_func,
                 init_method,
                 output_layer_init_method):
        super(TransformerLanguageModelIntermediateStage, self).__init__(
            attention_mask_func,
            init_method,
            output_layer_init_method)

    def forward(self, hidden_states, attention_mask,
                layer_past=None, get_key_value=False):
        return super(TransformerLanguageModelIntermediateStage, self).forward(
            hidden_states,
            attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value
        )


class TransformerLanguageModelLastStage(TransformerLanguageModelBase):
    """Transformer language model, final stage (see
       TransformerLanguageModelBase for description of arguments).
    """

    def __init__(self,
                 attention_mask_func,
                 init_method,
                 output_layer_init_method,
                 add_pooler=False):
        super(TransformerLanguageModelLastStage, self).__init__(
            attention_mask_func,
            init_method,
            output_layer_init_method,
            add_pooler=add_pooler)

    def forward(self, hidden_states, attention_mask,
                layer_past=None, get_key_value=False,
                pooling_sequence_index=0):
        return super(TransformerLanguageModelLastStage, self).forward(
            hidden_states,
            attention_mask,
            layer_past=layer_past,
            get_key_value=get_key_value,
            pooling_sequence_index=pooling_sequence_index
        )
