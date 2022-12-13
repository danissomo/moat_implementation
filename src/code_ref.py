import copy
import re
from typing import Optional, Any

import torch
import torch.nn as nn

# This handles the invalid name scope of tf.keras.sequential
# used in stem layers.
_STEM_LAYER_NAME_SCOPE = 'moat/stem/'

# This is for loading the exponential moving average of variables
# in the checkpoint.
_EMA_VARIABLE_NAME_POSTFIX = '/ExponentialMovingAverage'

# The position embedding size at stride 16 and 32.
# The input size changes, but the number of learnable parameters of position
# embedding does not change. The position embeddings are interpolated for
# different input sizes.
_STRIDE_16_POSITION_EMBEDDING_SIZE = 14
_STRIDE_32_POSITION_EMBEDDING_SIZE = 7


class MOAT(nn.Module):
  """MOAT backbone."""

  def _retrieve_config(self, config):
    """Retrieves the config of MOAT.
    Args:
      config: A dictionary containing the following keys.
        -stem_size: A list of integers, the list length is the number of stem
          layers, the list element is the output channels of the stem layer.
        -block_type: A list of strings, the list length is the stage number.
          The list element is the block type for a stage. The supported block
          types are 'mbconv' and 'moat'.
        -num_blocks: A list of integers, the list length is the stage number.
          The list element is the number of blocks in a stage.
        -hidden_size: A list of integers, the list length is the stage number.
          The list element is the output channels of the blocks in a stage.
        -stage_stride: A list of intergers, the stride of first block in
          each stage.
        -expansion_rate: An integer, expansion rate of MBConv.
        -se_ratio: An integer, expansion ratio of SE in MBConv block.
        -head_size: An integer, feature channels per head in Attention.
        -window_size: A list of list of integers, specifying the window size for
          each stage. The length should be the number of stages. For example:
          [None, None, [14, 16], [7, 8]] means the last two stages uses
          14 by 16 and 7 by 8 windows, respectively.
        -position_embedding_size: A list of integers, specifying the position
          embedding sizes for all stages. If the feature maps have larger sizes,
          the position embeddings will be interpolated to match them.
        -use_checkpointing_for_attention: A boolean, specifying whether to use
          checkpointing for attention.
        -global_attention_at_end_of_moat_stage: A boolean, specifying whether
          to use global attention for the last block if the stage consists of
          moat blocks.
        -relative_position_embedding_type: A string, type of relative position
          embedding in Attention. Currently, only '2d_multi_head' is supported.
          If None, no relative position embedding will be used.
        -ln_epsilon: A float, epsilon for layer normalization in Attention.
        -pool_size: An integer, kernel size for pooling in shortcut branch in
          MBConv block and MOAT block.
        -survival_prob: A float, 1 - drop_path_rate.
        -kernel_initializer: Initializer for the kernel weights matrix.
        -bias_initializer: Initializer for the bias vector.
        -name: A string, model name.
        -build_classification_head_with_class_num: An integer, number of
          classes. If None, no classification head will be built.
    Returns:
      A Config class: hparams_config.Config.
    Raises:
      ValueError: If the lengths of block_type, num_blocks and hidden_size are
        not the same.
      ValueError: If the element of block_type is not one of ['mbconv', 'moat'].
    """

    required_keys = ['stem_size', 'block_type', 'num_blocks', 'hidden_size']
    optional_keys = {
        'stage_stride': [2, 2, 2, 2],
        'expansion_rate': 4,
        'se_ratio': 0.25,
        'head_size': 32,
        'window_size': [None, None, [14, 14], [7, 7]],
        'position_embedding_size': [
            None, None,
            _STRIDE_16_POSITION_EMBEDDING_SIZE,
            _STRIDE_32_POSITION_EMBEDDING_SIZE],
        'use_checkpointing_for_attention': False,
        'global_attention_at_end_of_moat_stage': False,
        'relative_position_embedding_type': '2d_multi_head',
        'ln_epsilon': 1e-5,
        'pool_size': 2,
        'survival_prob': None,
        'kernel_initializer': tf.random_normal_initializer(stddev=0.02),
        'bias_initializer': tf.zeros_initializer,
        'build_classification_head_with_class_num': None,
    }
    config = create_config_from_dict(config, required_keys, optional_keys)

    stage_number = len(config.block_type)
    if (len(config.num_blocks) != stage_number or
        len(config.hidden_size) != stage_number):
      raise ValueError('The lengths of block_type, num_blocks and hidden_size ',
                       'should be the same.')
    return config

  def _local_config(self, config, idx, exclude_regex=None):
    """Gets stage-wise config from backbone-wise config."""

    config = copy.deepcopy(config)
    for key in config.__dict__:
      if isinstance(config[key], (list, tuple)):
        if exclude_regex is None or not re.search(exclude_regex, key):
          config[key] = config[key][idx]
    return config

  def __init__(self, **config):
    super().__init__(name='moat')
    self._config = self._retrieve_config(config)

  def _build_stem(self):
    stem_layers = []
    for i in range(len(self._config.stem_size)):
      conv_layer = tf.keras.layers.Conv2D(
          filters=self._config.stem_size[i],
          kernel_size=3,
          strides=2 if i == 0 else 1,
          padding='same',
          kernel_initializer=self._config.kernel_initializer,
          bias_initializer=self._config.bias_initializer,
          use_bias=True,
          name='conv_{}'.format(i))
      stem_layers.append(conv_layer)
      if i < len(self._config.stem_size) - 1:
        stem_layers.append(self._config.norm_class(name='norm_{}'.format(i)))
        stem_layers.append(tf.keras.layers.Activation(
            self._config.activation, name='act_{}'.format(i)))
    # The name scope of tf.keras.Sequential is invalid, see error handling
    # in the part of loading checkpoints in function get_model.
    self._stem = tf.keras.Sequential(
        layers=stem_layers,
        name='stem')

  def _build_block(self, local_block_config):
    if local_block_config.block_type == 'mbconv':
      block = MBConvBlock(**local_block_config)
    elif local_block_config.block_type == 'moat':
      block = MOATBlock(**local_block_config)
    else:
      raise ValueError('Unsupported block_type: {}'.format(
          local_block_config.block_type))
    return block

  def _build_classification_head(self):
    self._final_layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=self._config.ln_epsilon,
        name='final_layer_norm')
    self._logits_head = tf.keras.layers.Conv2D(
        filters=self._config.build_classification_head_with_class_num,
        kernel_size=1,
        strides=1,
        kernel_initializer=self._config.kernel_initializer,
        bias_initializer=self._config.bias_initializer,
        padding='same',
        use_bias=True,
        name='logits_head')

  def _adjust_survival_rate(self, local_block_config,
                            block_id, total_num_blocks):
    survival_prob = self._config.survival_prob
    if survival_prob is not None:
      drop_rate = 1.0 - survival_prob
      survival_prob = 1.0 - drop_rate * block_id / total_num_blocks
      local_block_config = local_block_config.replace(
          survival_prob=survival_prob)
    return local_block_config

  def build(self, input_shape: list[int]) -> None:
    norm_class = tf.keras.layers.experimental.SyncBatchNormalization
    self._config.norm_class = norm_class
    self._config.activation = tf.nn.gelu

    self._build_stem()

    self._blocks = []
    total_num_blocks = sum(self._config.num_blocks)

    for stage_id in range(len(self._config.block_type)):
      stage_config = self._local_config(self._config, stage_id, '^stem.*')
      stage_blocks = []

      for local_block_id in range(stage_config.num_blocks):
        local_block_config = copy.deepcopy(stage_config)

        block_stride = 1
        if local_block_id == 0:
          block_stride = self._config.stage_stride[stage_id]
        local_block_config = local_block_config.replace(
            block_stride=block_stride)

        block_id = sum(self._config.num_blocks[:stage_id]) + local_block_id
        local_block_config = self._adjust_survival_rate(
            local_block_config,
            block_id, total_num_blocks)

        block_name = 'block_{:0>2d}_{:0>2d}'.format(stage_id, local_block_id)
        local_block_config.name = block_name

        if (local_block_id == stage_config.num_blocks - 1 and
            self._config.block_type[stage_id] == 'moat' and
            self._config.global_attention_at_end_of_moat_stage):
          local_block_config.window_size = None

        block = self._build_block(local_block_config)
        stage_blocks.append(block)

      self._blocks.append(stage_blocks)

    if self._config.build_classification_head_with_class_num is not None:
      self._build_classification_head()

  def call(self, inputs, training=False, mask=None):
    endpoints = {}

    output = self._stem(inputs, training=training)
    endpoints['stage1'] = output
    endpoints['res1'] = self._config.activation(output)

    for stage_id, stage_blocks in enumerate(self._blocks):
      for block in stage_blocks:
        output = block(output, training=training)
      endpoints['stage{}'.format(stage_id + 2)] = output
      endpoints['res{}'.format(stage_id + 2)] = self._config.activation(output)

    if self._config.build_classification_head_with_class_num is None:
      return endpoints
    else:
      reduce_axes = list(range(1, output.shape.rank - 1))
      output = tf.reduce_mean(output, axis=reduce_axes, keepdims=True)
      output = self._final_layer_norm(output)
      output = self._logits_head(output, training=training)
      logits = tf.squeeze(output, axis=[1, 2])
      return logits
