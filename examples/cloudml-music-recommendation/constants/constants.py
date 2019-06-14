# Copyright 2019 Google Inc. All Rights Reserved.
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
"""Constants for preprocessing and training scripts."""

import tensorflow as tf


# file patterns
TRAIN_PATTERN = "train*.tfrecord"
EVAL_PATTERN = "eval*.tfrecord"
TEST_PATTERN = "test*.tfrecord"
USER_VOCAB_NAME = "vocab_users"
ITEM_VOCAB_NAME = "vocab_items"
ARTIST_VOCAB_NAME = "vocab_artists"
TAG_VOCAB_NAME = "vocab_tags"
TMP_DIR = "tmp"

# bq data
USER_KEY = "user"
ITEM_KEY = "song"
ARTIST_KEY = "artist"
ALBUMS_KEY = "albums"
TAGS_KEY = "tags"
COUNT_KEY = "count_norm"
TOP_10_KEY = "top_10"
BQ_FEATURES = [
    USER_KEY,
    ITEM_KEY,
    ARTIST_KEY,
    ALBUMS_KEY,
    TAGS_KEY,
    TOP_10_KEY,
    COUNT_KEY,
]

# tft data
TFT_USER_KEY = "tft_user"
TFT_ITEM_KEY = "tft_item"
TFT_ARTIST_KEY = "tft_artist"
TFT_TAGS_KEY = "tft_tags"
TFT_TOP_10_KEY = "tft_top_10"
TFT_FEATURES = [
    TFT_USER_KEY,
    TFT_ITEM_KEY,
    TFT_ARTIST_KEY,
    TFT_TAGS_KEY,
    TFT_TOP_10_KEY,
]
TFT_DEFAULT_ID = -1

TRAIN_SIZE = .7
EVAL_SIZE = .15


def _get_train_spec():
  """Returns a dict mapping training features to tfrecord features."""
  train_spec = {}
  for key in [USER_KEY, ITEM_KEY, ARTIST_KEY]:
    train_spec[key] = tf.FixedLenFeature([], dtype=tf.string)
  for key in [ALBUMS_KEY]:
    train_spec[key] = tf.FixedLenFeature([], dtype=tf.int64)
  for key in [COUNT_KEY]:
    train_spec[key] = tf.FixedLenFeature([], dtype=tf.float32)
  for key in [TAGS_KEY, TOP_10_KEY]:
    train_spec[key] = tf.VarLenFeature(tf.string)
  return train_spec


def get_serving_stub():
  """Returns stubbed values for features to use during serving when only username matters."""
  stub = {}
  for key in [USER_KEY, ITEM_KEY, ARTIST_KEY]:
    stub[key] = ""
  for key in [ALBUMS_KEY]:
    stub[key] = 0
  for key in [TAGS_KEY, TOP_10_KEY]:
    stub[key] = []
  return stub


# model constants
USER_NUMERICAL_FEATURES = []
USER_CATEGORICAL_FEATURES = [TFT_TOP_10_KEY]
USER_CATEGORICAL_VOCABS = [ITEM_VOCAB_NAME]
USER_FEATURES = USER_NUMERICAL_FEATURES + USER_CATEGORICAL_FEATURES
ITEM_NUMERICAL_FEATURES = [ALBUMS_KEY]
ITEM_CATEGORICAL_FEATURES = [TFT_TAGS_KEY]
ITEM_CATEGORICAL_VOCABS = [TAG_VOCAB_NAME]
ITEM_FEATURES = ITEM_NUMERICAL_FEATURES + ITEM_CATEGORICAL_FEATURES

EVAL_RECALLS = [10, 100, 1000]
TRAIN_SPEC = _get_train_spec()
SERVE_SPEC = _get_train_spec()
SERVE_SPEC.pop(COUNT_KEY)
RAW_CATEGORICAL_FEATURES = [TAGS_KEY, TOP_10_KEY]

# tensorboard projector config
PROJECTOR_PATH = "metadata.tsv"
PROJECTOR_NAME = "combined_embedding"
PROJECTOR_USER_SAMPLES = 1000
PROJECTOR_ITEM_SAMPLES = 4000
