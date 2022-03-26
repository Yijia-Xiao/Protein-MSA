# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2021, Knowledge Engineering Group (KEG), Tsinghua University
# Modified by Jiezhong Qiu
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

"""Pretrain MSA"""

import torch
import torch.nn.functional as F

from megatron import get_args, get_tokenizer
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.msa_dataset import build_train_valid_test_datasets
from megatron.model import MSAModel, MSAModelFirstStage, MSAModelIntermediateStage, MSAModelLastStage
from megatron.model.transformer import Collector
from megatron.training import pretrain
from megatron.utils import average_losses_across_data_parallel_group
from megatron.utils import get_msa_masks_and_position_ids

from megatron.model.msa_model import bert_extended_attention_mask
from megatron import IterCounter

def model_provider():
    """Build the model."""

    print_rank_0('building MSA model ...')

    args = get_args()
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        # Determine model based on position of stage in pipeline.
        if mpu.is_pipeline_first_stage():
            model = MSAModelFirstStage(
                num_tokentypes=0)
        elif mpu.is_pipeline_last_stage():
            model = MSAModelLastStage(
                num_tokentypes=0,
                add_binary_head=False,
                parallel_output=True)
        else:
            model = MSAModelIntermediateStage(
                num_tokentypes=0)
    else:
        model = MSAModel(
            num_tokentypes=0,
            add_binary_head=False,
            parallel_output=True)

    return model

def tokens_to_seq(alig):
    msa_vocab = {0: '[PAD]', 1: '[MASK]', 2: '[CLS]', 3: '[SEP]', 4: '[UNK]', 5: 'A', 6: 'B', 7: 'C', 8: 'D', 9: 'E', 10: 'F', 11: 'G', 12: 'H', 13: 'I', 14: 'K', 15: 'L', 16: 'M', 17: 'N', 18: 'O', 19: 'P', 20: 'Q', 21: 'R', 22: 'S', 23: 'T', 24: 'U', 25: 'V', 26: 'W', 27: 'X', 28: 'Y', 29: 'Z', 30: '-', 31: '|'}
    # seq = [''.join([msa_vocab[idx.item()] for idx in alig]) for alig in raw_msa_sample]
    seq = ''.join([msa_vocab[idx.item()] for idx in alig])
    return seq


def get_batch(data_iterator):
    """Build the batch."""

    args = get_args()
    tokenizer = get_tokenizer()
    # Items and their type.
    keys = ['text', 'labels', 'loss_mask', 'offset', 'msa_aligns', 'msa_length', 'raw_msa_sample'] # , 'padding_mask']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    # TODO: support protein string return
    # data, seq = data
    # data, msa_shape, seq = data
    data_b = mpu.broadcast_data(keys, data, datatype)


    # Unpack.
    tokens = data_b['text'].long()[0]
    loss_mask = data_b['loss_mask'].float()[0]
    lm_labels = data_b['labels'].long()[0]
    offset = data_b['offset'].long()[0]
    msa_aligns = data_b['msa_aligns'].long()[0]
    msa_length = data_b['msa_length'].long()[0]
    raw_msa_sample = data_b['raw_msa_sample'].long()[0]
    msa_shape = (msa_aligns, msa_length)
    # padding_mask = data_b['padding_mask'].long()[0]

    # Get the masks and postition ids.
    # micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    # if reset_attention_mask:
    #     att_mask_batch = micro_batch_size
    # else:
    #     att_mask_batch = 1

    # attention_mask = torch.ones(
    #     (att_mask_batch, seq_length, seq_length), device=data.device).view(
    #         att_mask_batch, 1, seq_length, seq_length)

    # Position ids.
    # seq_aligns, seq_length = msa_shape
    # TODO: well done debug: here I can found the bug in offset -1 (cause insertion of [CLS]), max_offset should be 256, not 257
    # print(f'{msa_shape[1].item()=}, {offset=}')
    position_ids = torch.arange(msa_shape[1].item(), dtype=torch.long,
                                device=tokens.device) + offset
    position_ids[0] = 0
    # print(f'{position_ids=}')
    if args.fake_input:
        position_ids += 2
    position_ids = position_ids.unsqueeze(0).expand_as(tokens)
    # position_ids = position_ids


    # TODO: position_ids + 2
    # if get_args().fake_input:
    #     position_ids += 2
    # position_ids = (torch.arange(msa_shape[1].item(), dtype=torch.long,
    #                             device=tokens.device) + 2).unsqueeze(0).expand_as(tokens)

    # return tokens, loss_mask, lm_labels, padding_mask, attention_mask, position_ids # , seq
    # print(f'{tokens=}, {loss_mask=}, {lm_labels=}, {position_ids=}')
    seq = tokens_to_seq(raw_msa_sample[0]) if args.attention_save else []
    return tokens, loss_mask, lm_labels, position_ids, seq


def forward_step(data_iterator, model, input_tensor):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    # TODO: support protein string return
    # tokens, loss_mask, lm_labels, padding_mask, attention_mask, position_ids, seq \
    tokens, loss_mask, lm_labels, position_ids, seq \
        = get_batch(data_iterator)
    timers('batch-generator').stop()
    # print_rank_0('in-pretrain_msa.py get... {}'.format(IterCounter.get_iter()))

    # extended_attention_mask = bert_extended_attention_mask(padding_mask) + attention_mask

    # Forward pass through the model.
    if mpu.is_pipeline_first_stage():
        assert input_tensor is None
        if mpu.is_pipeline_last_stage():
            if args.attention_save:
                # if tokens.shape[1] > 1023:
                eval_max_length = args.eval_max_length
                print(f'len={tokens.shape[1]}')
                if tokens.shape[1] > eval_max_length:
                    print(f'skipping one sample longer than {eval_max_length}, len={tokens.shape[1]}')
                    return 0, {'lm loss': 0}
                # NOTICE: remember to change return function of `get_batch` function
                Collector.append(seq)
            output_tensor = model(tokens, tokentype_ids=None,
                                  lm_labels=lm_labels, position_ids=position_ids)
        else:
            output_tensor = model(tokens, tokentype_ids=None)
    elif mpu.is_pipeline_last_stage():
        assert input_tensor is not None
        output_tensor = model(input_tensor, lm_labels=lm_labels)
    else:
        assert input_tensor is not None
        output_tensor = model(input_tensor, position_ids=position_ids)

    if mpu.is_pipeline_last_stage():
        lm_loss_, _ = output_tensor

        lm_loss_ = lm_loss_.float()
        loss_mask = loss_mask.float()
        lm_loss = torch.sum(
            lm_loss_.view(-1) * loss_mask.reshape(-1)) / loss_mask.sum()

        loss = lm_loss

        averaged_losses = average_losses_across_data_parallel_group([lm_loss,])

        return loss, {'lm loss': averaged_losses[0]}
    return output_tensor


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for MSA ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        masked_lm_prob=args.mask_prob,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating MSA datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})
    if get_args().attention_save:
        Collector.dump('./data/attention')
