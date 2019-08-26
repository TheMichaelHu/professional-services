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


def _model_fn():
  """A model function for item recommendation."""

  model = tf.keras.Sequential()
  model.add(tf.keras.layers.LSTM(100, activation="relu", input_shape=(n_in,1)))
  model.add(tf.keras.layers.RepeatVector(n_in))
  model.add(tf.keras.layers.LSTM(100, activation="relu", return_sequences=True))
  model.add(tf.keras.layers.TimeDistributed(Dense(1)))

  model.compile(optimizer="adam", loss="mse")
  return model


def _get_trial_id():
  """Returns the trial id if it exists, else "0"."""
  trial_id = json.loads(
      os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial", "")
  return trial_id if trial_id else "1"


def get_model(params):
  """Returns the product recommendation model."""
  config = tf.estimator.RunConfig(
      save_checkpoints_steps=params.save_checkpoints_steps,
      keep_checkpoint_max=params.keep_checkpoint_max,
      log_step_count_steps=params.log_step_count_steps)
  trial_id = _get_trial_id()
  model_dir = os.path.join(params.model_dir, trial_id)

  hparams = tf.contrib.training.HParams(
      learning_rate=params.learning_rate,
      num_layers=params.num_layers,
      embedding_size=params.embedding_size)
  model_params = {
      "model_dir": model_dir,
      "hparams": hparams,
  }
  estimator = tf.keras.estimator.model_to_estimator(
      keras_model=model_fn(),
      model_dir=model_dir,
      config=config,
      params=model_params)
  return estimator

