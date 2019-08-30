# Copyright 2019 Google Inc. All Rights Reserved.

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
"""Defines the model for product recommendation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

import tensorflow as tf

# pylint: disable=g-bad-import-order
from constants import constants


def _make_model(hparams):
  return tf.keras.Sequential([
      tf.keras.layers.LSTM(hparams.encoding_dims),
      tf.keras.layers.RepeatVector(constants.WINDOW_SIZE),
      tf.keras.layers.LSTM(hparams.decoding_dims, return_sequences=True),
      tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1)),
  ])


# pylint: disable=unused-argument
def _model_fn(features, labels, mode, params):
  """."""
  hparams = params["hparams"]
  model = _make_model(hparams)

  # Stack features to make 3D Tensors of shape (batch, time, features)
  seq = tf.stack([features[x] for x in constants.FEATURES], axis=2)
  training = (mode == tf.estimator.ModeKeys.TRAIN)
  predictions = model(seq, training=training)

  # Predicting the reverse of the sequence is often easier.
  predictions = tf.reverse(tf.squeeze(predictions), [1])
  predictions = tf.reverse_sequence(
      tf.squeeze(predictions),
      seq_lengths=tf.tile(
          tf.constant([constants.WINDOW_SIZE], dtype=tf.int64),
          multiples=tf.expand_dims(hparams.batch_size, axis=0)),
      seq_axis=1,
      batch_axis=0)

  if mode == tf.estimator.ModeKeys.PREDICT:
    prediction_out = {
        "predictions": predictions,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=prediction_out)

  loss = tf.losses.mean_squared_error(features[constants.MEASUREMENT_KEY],
                                      predictions)

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss)

  # Training op: Update the weights via backpropagation.
  optimizer = tf.train.AdagradOptimizer(learning_rate=hparams.learning_rate)
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def _get_trial_id():
  """Returns the trial id if it exists, else "0"."""
  trial_id = json.loads(
      os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial", "")
  return trial_id if trial_id else "1"


def get_model(params):
  """."""
  config = tf.estimator.RunConfig(
      save_checkpoints_steps=params.save_checkpoints_steps,
      keep_checkpoint_max=params.keep_checkpoint_max,
      log_step_count_steps=params.log_step_count_steps)
  trial_id = _get_trial_id()
  model_dir = os.path.join(params.model_dir, trial_id)

  hparams = tf.contrib.training.HParams(
      batch_size=params.batch_size,
      learning_rate=params.learning_rate,
      encoding_dims=params.encoding_dims,
      decoding_dims=params.decoding_dims,
  )
  model_params = {
      "model_dir": model_dir,
      "hparams": hparams,
  }
  estimator = tf.estimator.Estimator(
      model_fn=_model_fn,
      model_dir=model_dir,
      config=config,
      params=model_params)
  return estimator

