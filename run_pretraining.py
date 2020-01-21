# coding=utf-8
# Copyright 2018 The Google AI Team Authors.
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
# Lint as: python2, python3
"""Run masked LM/next sentence masked_lm pre-training for ALBERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import modeling
import optimization
from six.moves import range
import tensorflow as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import data as contrib_data
from tensorflow.contrib import tpu as contrib_tpu
# from tf.estimator import BestExporter, EvalSpec
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "albert_config_file", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained ALBERT model).")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 4096, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 64, "Total batch size for eval.")

flags.DEFINE_enum("optimizer", "lamb", ["adamw", "lamb"],
                  "The optimizer for training.")

flags.DEFINE_float("learning_rate", 0.00176, "The initial learning rate.")

flags.DEFINE_float("poly_power", 1.0, "The power of poly decay.")

flags.DEFINE_integer("num_train_steps", 125000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 3125, "Number of warmup steps.")

flags.DEFINE_integer("start_warmup_step", 0, "The starting step of warmup.")

flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_max", 20,
                     "How many checkpoints to keep.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", True, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("init_from_group0", False, "Whether to initialize"
                  "parameters of other groups from group 0")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_float(
    "masked_lm_budget", 0,
    "If >0, the ratio of masked ngrams to unmasked ngrams. Default 0,"
    "for offline masking")

flags.DEFINE_bool(
    "do_hydro", False,
    "Whether or not to use local hydrophobicity predictions in training.")

flags.DEFINE_bool(
    "do_charge", False,
    "Whether or not to use local charge predictions in training.")

flags.DEFINE_bool(
    "do_pks", False,
    "Whether or not to use local predictions of pKa NH2, pKa COOH in training.")

flags.DEFINE_bool(
    "do_solubility", False,
    "Whether or not to use local predictions of solubility in training.")




def model_fn_builder(albert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, optimizer, poly_power,
                     start_warmup_step, 
                     do_hydro=False, do_charge=False, do_pks=False, do_solubility=False):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    hydrophobicities = features["hydrophobicities"]
    solubilities = features["solubilities"]
    charges = features["charges"]
    pks = features["pks"]
    masked_lm_weights = features["masked_lm_weights"]
    hydrophobicity_weights = features["hydrophobicity_weights"]
    solubility_weights = features["solubility_weights"]
    charge_weights = features["charge_weights"]
    pk_weights = features["pk_weights"]


    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    k = albert_config.k #!NOTE: this is the length of the kmer

    model = modeling.AlbertModel(
        config=albert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    (masked_lm_loss, masked_lm_example_loss,
     masked_lm_log_probs) = get_masked_lm_output(albert_config,
                                                 model.get_sequence_output(),
                                                 model.get_embedding_table(),
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights)

    if do_hydro:
      (hydrophobicity_loss, hydrophobicity_example_loss, hydrophobicity_log_probs) = get_hydrophobicity_output(
          albert_config, model.get_sequence_output(),
          masked_lm_positions, hydrophobicities, hydrophobicity_weights)
    else:
      (hydrophobicity_loss, hydrophobicity_example_loss, hydrophobicity_log_probs) = (0, 0, None)

    if do_charge:
      (charge_loss, charge_example_loss, charge_log_probs) = get_charge_output(
          albert_config, model.get_sequence_output(), 
          masked_lm_positions, charges, charge_weights)
    else:
      (charge_loss, charge_example_loss, charge_log_probs) = (0, 0, None)

    if do_pks:
      (pk_loss, pk_example_loss, pk_log_probs) = get_pk_output(
          albert_config, model.get_sequence_output(), 
          masked_lm_positions, pks, pk_weights)
    else:
      (pk_loss, pk_example_loss, pk_log_probs) = (0, 0, None)

    if do_solubility:
      (solubility_loss, solubility_example_loss, solubility_log_probs) = get_solubility_output(
          albert_config, model.get_sequence_output(),
          masked_lm_positions, solubilities, solubility_weights)
    else:
      (solubility_loss, solubility_example_loss, solubility_log_probs) = (0, 0, None)

    total_loss = masked_lm_loss + hydrophobicity_loss + charge_loss + pk_loss + solubility_loss

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      tf.logging.info("number of hidden group %d to initialize",
                      albert_config.num_hidden_groups)
      num_of_initialize_group = 1
      if FLAGS.init_from_group0:
        num_of_initialize_group = albert_config.num_hidden_groups
        if albert_config.net_structure_type > 0:
          num_of_initialize_group = albert_config.num_hidden_layers
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(
              tvars, init_checkpoint, num_of_initialize_group)
      if use_tpu:

        def tpu_scaffold():
          for gid in range(num_of_initialize_group):
            tf.logging.info("initialize the %dth layer", gid)
            tf.logging.info(assignment_map[gid])
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map[gid])
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        for gid in range(num_of_initialize_group):
          tf.logging.info("initialize the %dth layer", gid)
          tf.logging.info(assignment_map[gid])
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map[gid])

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps,
          use_tpu, optimizer, poly_power, start_warmup_step)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(*args):
        """Computes the loss and accuracy of the model."""
        (masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights, 
            hydrophobicity_example_loss, hydrophobicity_log_probs, hydrophobicities, hydrophobicity_weights,
            charge_example_loss, charge_log_probs, charges, charge_weights,
            pk_example_loss, pk_log_probs, pks, pk_weights,
            solubility_example_loss, solubility_log_probs, solubilities, solubility_weights) = args[:20]

        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        if do_hydro:
          hydrophobicity_log_probs = tf.reshape(hydrophobicity_log_probs,
                                          [-1, hydrophobicity_log_probs.shape[-1]])
          hydrophobicity_predictions = tf.argmax(
              hydrophobicity_log_probs, axis=-1, output_type=tf.int32)
          hydrophobicity_example_loss = tf.reshape(hydrophobicity_example_loss, [-1])
          hydrophobicities = tf.reshape(hydrophobicities, [-1])
          hydrophobicity_weights = tf.reshape(hydrophobicity_weights, [-1])
          hydrophobicity_accuracy = tf.metrics.accuracy(
              labels=hydrophobicities,
              predictions=hydrophobicity_predictions,
              weights=hydrophobicity_weights)
          hydrophobicity_mean_loss = tf.metrics.mean(
              values=hydrophobicity_example_loss, weights=hydrophobicity_weights)
        else:
          hydrophobicity_accuracy = 0
          hydrophobicity_mean_loss = 0

        if do_charge:
          charge_log_probs = tf.reshape(charge_log_probs,
                                          [-1, charge_log_probs.shape[-1]])
          charge_predictions = tf.argmax(
              charge_log_probs, axis=-1, output_type=tf.int32)
          charge_example_loss = tf.reshape(charge_example_loss, [-1])
          charges = tf.reshape(charges, [-1])
          charge_weights = tf.reshape(charge_weights, [-1])
          charge_accuracy = tf.metrics.accuracy(
              labels=charges,
              predictions=charge_predictions,
              weights=charge_weights)
          charge_mean_loss = tf.metrics.mean(
              values=charge_example_loss, weights=charge_weights)
        else:
          charge_accuracy = 0
          charge_mean_loss = 0

        if do_pks:
          pk_log_probs = tf.reshape(pk_log_probs,
                                          [-1, pk_log_probs.shape[-1]])
          pk_predictions = tf.argmax(
              pk_log_probs, axis=-1, output_type=tf.int32)
          pk_example_loss = tf.reshape(pk_example_loss, [-1])
          pks = tf.reshape(pks, [-1])
          pk_weights = tf.reshape(pk_weights, [-1])
          pk_accuracy = tf.metrics.accuracy(
              labels=pks,
              predictions=pk_predictions,
              weights=pk_weights)
          pk_mean_loss = tf.metrics.mean(
              values=pk_example_loss, weights=pk_weights)
        else:
          pk_accuracy = 0
          pk_mean_loss = 0

        if do_solubility:
          solubility_log_probs = tf.reshape(solubility_log_probs,
                                          [-1, solubility_log_probs.shape[-1]])
          solubility_predictions = tf.argmax(
              solubility_log_probs, axis=-1, output_type=tf.int32)
          solubility_example_loss = tf.reshape(solubility_example_loss, [-1])
          solubilities = tf.reshape(solubilities, [-1])
          solubility_weights = tf.reshape(solubility_weights, [-1])
          solubility_accuracy = tf.metrics.accuracy(
              labels=solubilities,
              predictions=solubility_predictions,
              weights=solubility_weights)
          solubility_mean_loss = tf.metrics.mean(
              values=solubility_example_loss, weights=solubility_weights)
        else:
          solubility_accuracy = 0
          solubility_mean_loss = 0

        metrics = {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "hydrophobicity_accuracy": hydrophobicity_accuracy,
            "hydrophobicity_loss": hydrophobicity_mean_loss,
            "charge_accuracy": charge_accuracy,
            "charge_loss": charge_mean_loss,
            "pk_accuracy": pk_accuracy,
            "pk_loss": pk_mean_loss,
            "solubility_accuracy": solubility_accuracy,
            "solubility_loss": solubility_mean_loss
        }

        return metrics

      metric_values = [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids, masked_lm_weights, 
          hydrophobicity_example_loss, hydrophobicity_log_probs, hydrophobicities, hydrophobicity_weights,
          charge_example_loss, charge_log_probs, charges, charge_weights,
          pk_example_loss, pk_log_probs, pks, pk_weights,
          solubility_example_loss, solubility_log_probs, solubilities, solubility_weights
      ]

      eval_metrics = (metric_fn, metric_values)

      output_spec = contrib_tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn


def get_masked_lm_output(albert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)


  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=albert_config.embedding_size,
          activation=modeling.get_activation(albert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              albert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[albert_config.vocab_size],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=albert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)



def get_hydrophobicity_output(bert_config, input_tensor, positions,
                         label_hydrophobicities, label_weights):
  """Get loss and log probs for the hydrophobicity prediction."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/hydrophobicity"):
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

      output_weights = tf.get_variable(
        "output_weights",
        shape=[3, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
      output_bias = tf.get_variable(
          "output_bias",
        shape=[3],
        initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_hydrophobicities = tf.reshape(label_hydrophobicities, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(label_hydrophobicities, depth=3, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_charge_output(bert_config, input_tensor, positions,
                         label_charges, label_weights):
  """Get loss and log probs for the charge prediction."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/charge"):
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    output_weights = tf.get_variable(
        "output_weights",
        shape=[3, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias",
        shape=[3],
        initializer=tf.zeros_initializer())

    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_charges = tf.reshape(label_charges, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(label_charges, depth=3, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_pk_output(bert_config, input_tensor, positions,
                         label_pks, label_weights):
  """Get loss and log probs for the pk prediction."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/pks"):
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    output_weights = tf.get_variable(
        "output_weights",
        shape=[3, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias",
        shape=[3],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_pks = tf.reshape(label_pks, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(label_pks, depth=3, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)



def get_solubility_output(bert_config, input_tensor, positions,
                         label_solubilities, label_weights, log=False):
  """Get loss and log probs for the solubility prediction."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/solubility"):
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    output_weights = tf.get_variable(
        "output_weights",
        shape=[3, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias",
        shape=[3],
        initializer=tf.zeros_initializer())
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_solubilities = tf.reshape(label_solubilities, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(label_solubilities, depth=3, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([max_seq_length], tf.int64),
        "hydrophobicities": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "solubilities": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "charges": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "pks": tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "hydrophobicity_weights": tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "pk_weights": tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "charge_weights": tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "solubility_weights": tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
    }

    if FLAGS.masked_lm_budget:
      name_to_features.update({
          "token_boundary":
              tf.FixedLenFeature([max_seq_length], tf.int64)})
    else:
      name_to_features.update({
          "masked_lm_positions":
              tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
          "masked_lm_ids":
              tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
          "masked_lm_weights":
              tf.FixedLenFeature([max_predictions_per_seq], tf.float32)})

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          contrib_data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.data.experimental.map_and_batch_with_legacy_function(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    tf.logging.info(d)
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  albert_config = modeling.AlbertConfig.from_json_file(FLAGS.albert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = contrib_cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
  run_config = contrib_tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      keep_checkpoint_max=FLAGS.keep_checkpoint_max,
      tpu_config=contrib_tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  model_fn = model_fn_builder(
      albert_config=albert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      optimizer=FLAGS.optimizer,
      poly_power=FLAGS.poly_power,
      start_warmup_step=FLAGS.start_warmup_step,
      do_hydro=FLAGS.do_hydro,
      do_charge=FLAGS.do_charge,
      do_pks=FLAGS.do_pks,
      do_solubility=FLAGS.do_solubility)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = contrib_tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)

    # ~TODO:finish
    # exporter = BestExporter(
    #       compare_fn=_loss_smaller,
    #       exports_to_keep=5)

    # eval_spec = EvalSpec(
    #     train_input_fn,
    #     steps,
    #     exporters)
    
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps, eval_spec=eval_spec)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    global_step = -1
    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    writer = tf.gfile.GFile(output_eval_file, "w")
    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)
    best_perf = 0
    key_name = "masked_lm_accuracy"
    while global_step < FLAGS.num_train_steps:
      if estimator.latest_checkpoint() is None:
        tf.logging.info("No checkpoint found yet. Sleeping.")
        time.sleep(1)
      else:
        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)
        global_step = result["global_step"]
        tf.logging.info("***** Eval results *****")
        checkpoint_path = estimator.latest_checkpoint()
        for key in sorted(result.keys()):
          tf.logging.info("  %s = %s", key, str(result[key]))
          writer.write("%s = %s\n" % (key, str(result[key])))
          if result[key_name] > best_perf:
            best_perf = result[key_name]
            for ext in ["meta", "data-00000-of-00001", "index"]:
              src_ckpt = checkpoint_path + ".{}".format(ext)
              tgt_ckpt = checkpoint_path.rsplit(
                  "-", 1)[0] + "-best.{}".format(ext)
              tf.logging.info("saving {} to {}".format(src_ckpt, tgt_ckpt))
              tf.gfile.Copy(src_ckpt, tgt_ckpt, overwrite=True)
              writer.write("saved {} to {}\n".format(src_ckpt, tgt_ckpt))


def custom_metric(labels, predictions):
    # This function will be called by the Estimator, passing its predictions.
    # Let's suppose you want to add the "mean" metric...

    # Accessing the class predictions (careful, the key name may change from one canned Estimator to another)
    predicted_classes = predictions["class_ids"]  

    # Defining the metric (value and update tensors):
    custom_metric = tf.metrics.mean(labels, predicted_classes, name="custom_metric")

    # Returning as a dict:
    return {"custom_metric": custom_metric}

if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("albert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
