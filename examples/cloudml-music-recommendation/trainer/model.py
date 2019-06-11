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
import tensorflow_transform as tft

from constants import constants  # pylint: disable=g-bad-import-order


def _default_embedding_size(vocab_size):
  """Returns a good dimension for an embedding given the vocab size.

  The 4th root of the number of categories is a good rule of thumb for choosing
  an embedding dimension according to:
  https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html

  Args:
    vocab_size: the number of categories being embedded.

  Returns:
    A good embedding dimension to use for the given vocab size.
  """
  return int(vocab_size**.25)


def _make_embedding_col(feature_name, vocab_name, tft_output, mult=1):
  """Creates an embedding column.

  Args:
    feature_name: a attribute of features to get embedding vectors for.
    vocab_name: the name of the embedding vocabulary made with tft.
    tft_output: a TFTransformOutput object.
    mult: a multiplier on the embedding size.

  Returns:
    A tuple of (embedding_col, embedding_size):
      embedding_col: an n x d tensor, where n is the batch size and d is the
        length of all the features concatenated together.
      embedding_size: the embedding dimension.
  """
  vocab_size = tft_output.vocabulary_size_by_name(vocab_name)
  embedding_size = int(_default_embedding_size(vocab_size) * mult)
  cat_col = tf.feature_column.categorical_column_with_identity(
      key=feature_name, num_buckets=vocab_size + 1, default_value=vocab_size)
  embedding_col = tf.feature_column.embedding_column(cat_col, embedding_size)
  return embedding_col, embedding_size


def _get_net_features(features, tft_output, n_feats, c_feats, vocabs):
  """Creates an input layer of features.

  Args:
    features: a batch of features.
    tft_output: a TFTransformOutput object.
    n_feats: a list of numerical feature names.
    c_feats: a list of categorical feature names.
    vocabs: a list of vocabulary names cooresponding the the features in
      c_feats.

  Returns:
    A tuple of (net_features, size):
      net_features: an n x d tensor, where n is the batch size and d is the
        length of all the features concatenated together.
      size: the size of the feature layer.
  """
  numerical_cols = [tf.feature_column.numeric_column(col) for col in n_feats]
  categorical_cols = [_make_embedding_col(col, vocab_name, tft_output)
                      for col, vocab_name in zip(c_feats, vocabs)]
  cols = [x[0] for x in categorical_cols] + numerical_cols
  size = sum([x[1] for x in categorical_cols]) + len(numerical_cols)
  feature_names = {x: features[x] for x in n_feats + c_feats}
  net_features = tf.feature_column.input_layer(feature_names, cols)
  return net_features, size


def _make_input_layer(features, tft_output, feature_name, vocab_name, n_feats,
                      c_feats, vocabs, mult=1):
  """Creates an input layer containing embeddings and features.

  Args:
    features: a batch of features.
    tft_output: a TFTransformOutput object.
    feature_name: a attribute of features to get embedding vectors for.
    vocab_name: the name of the embedding vocabulary made with tft.
    n_feats: a list of numerical feature names.
    c_feats: a list of categorical feature names.
    vocabs: a list of vocabulary names cooresponding the the features in
      c_features.
    mult: a multiplier on the embedding size.

  Returns:
    A tuple of (net, size):
      net: an n x d tensor, where n is the batch size and d is the embedding
        size.
      size: the size of the layer.
  """
  col, embedding_size = _make_embedding_col(feature_name, vocab_name,
                                            tft_output, mult)
  embedding_feature = tf.feature_column.input_layer(
      {feature_name: features[feature_name]}, [col])
  net_features, size = _get_net_features(features, tft_output, n_feats,
                                         c_feats, vocabs)
  net = tf.concat([embedding_feature, net_features], 1)
  return net, embedding_size + size


def _resize_networks(user_net, user_size, item_net, item_size, num_layers):
  """Use hidden layers to make the user and item embeddings the same size.

  Args:
    user_net: a tensor consisting of a user_id embedding and features.
    user_size: the size of the user_net layer.
    item_net: a tensor consisting of an item_id embedding and features.
    item_size: the size of the item_net layer.
    num_layers: the number of hidden layers to use for resizing.

  Returns:
    A tuple of (user_net, item_net):
      user_net: a tensor consisting of a user embedding.
      item_net: a tensor consisting of an item embedding.
  """
  embedding_size = min(user_size, item_size)
  layer_step_size = abs(user_size - item_size) // num_layers
  if user_size > item_size:
    for i in reversed(range(num_layers)):
      dims = i * layer_step_size + embedding_size
      user_net = tf.keras.layers.Dense(dims, activation="relu")(user_net)
  elif item_size > user_size:
    for i in reversed(range(num_layers)):
      dims = i * layer_step_size + embedding_size
      item_net = tf.keras.layers.Dense(dims, activation="relu")(item_net)

  # Use linear activations for the final layer
  user_net = tf.keras.layers.Dense(embedding_size)(user_net)
  item_net = tf.keras.layers.Dense(embedding_size)(item_net)
  return user_net, item_net


def _get_embedding_matrix(embedding_size, tft_output, vocab_name):
  """Returns a num_items x embedding_size lookup table of embeddings."""
  vocab_size = tft_output.vocabulary_size_by_name(vocab_name)
  return tf.get_variable(
      "{}_embedding".format(vocab_name),
      (vocab_size, embedding_size),
      initializer=tf.zeros_initializer())


def _update_embedding_matrix(row_indices, rows, embedding_size, tft_output,
                             vocab_name):
  """Creates and maintains a lookup table of embeddings for inference.

  Args:
    row_indices: indices of rows of the lookup table to update.
    rows: the values to update the lookup table with.
    embedding_size: the size of the embedding.
    tft_output: a TFTransformOutput object.
    vocab_name: a tft vocabulary name.

  Returns:
    A num_items x embedding_size table of the latest embeddings with the given
      rows updated.
  """
  embedding = _get_embedding_matrix(embedding_size, tft_output, vocab_name)
  return tf.scatter_update(embedding, row_indices, rows)


def _normalize(t):
  """Turn each row of the given tensor into a unit vector."""
  norm = tf.sqrt(tf.reduce_sum(tf.square(t), axis=1, keepdims=True))
  return t / norm


def _get_top_k(features, embedding, feature_name, item_embedding, k=10):
  """Get the k most similar items for a given feature (user or item).

  Args:
    features: a batch of features.
    embedding: an embedding matrix.
    feature_name: the name of the feature.
    item_embedding: an item embedding matrix.
    k: the number of similar items to return.

  Returns:
    A tuple of (similarities, items):
      similarities: the similarity score for each item.
      items: the k most similar items.
  """
  tft_ids = features[feature_name]
  indices = tf.where(
      tf.equal(tft_ids, constants.TFT_DEFAULT_ID),
      tf.zeros_like(tft_ids), tft_ids)
  norm = tf.gather(embedding, indices)
  sims = tf.matmul(norm, item_embedding, transpose_b=True)
  return tf.math.top_k(sims, k)


def _model_fn(features, labels, mode, params):
  """A model function for item recommendation.

  Two Tower Architecture:
  Builds neural nets for users and items that learn n-dimensional
  representations of each. The distance between these representations is used
  to make a prediction for a binary classification.

  Args:
    features: a batch of features.
    labels: a batch of labels or None if predicting.
    mode: an instance of tf.estimator.ModeKeys.
    params: a dict of additional params.

  Returns:
    A tf.estimator.EstimatorSpec that fully defines the model that will be run
      by an Estimator.
  """
  tft_output = tft.TFTransformOutput(params["tft_dir"])
  tft_features = tft_output.transform_raw_features(features)
  hparams = params["hparams"]

  # Build user and item networks.
  user_net, user_size = _make_input_layer(tft_features,
                                          tft_output,
                                          constants.TFT_USER_KEY,
                                          constants.USER_VOCAB_NAME,
                                          constants.USER_NUMERICAL_FEATURES,
                                          constants.USER_CATEGORICAL_FEATURES,
                                          constants.USER_CATEGORICAL_VOCABS,
                                          hparams.user_embed_mult)
  item_net, item_size = _make_input_layer(tft_features,
                                          tft_output,
                                          constants.TFT_ITEM_KEY,
                                          constants.ITEM_VOCAB_NAME,
                                          constants.ITEM_NUMERICAL_FEATURES,
                                          constants.ITEM_CATEGORICAL_FEATURES,
                                          constants.ITEM_CATEGORICAL_VOCABS,
                                          hparams.item_embed_mult)
  user_net, item_net = _resize_networks(user_net, user_size, item_net,
                                        item_size, hparams.num_layers)
  embedding_size = min(user_size, item_size)

  # Map cosine similarity of user and item embeddings to predicted listen count.
  user_norm = _normalize(user_net)
  item_norm = _normalize(item_net)

  # Prediction op
  if mode == tf.estimator.ModeKeys.PREDICT:
    table = tf.contrib.lookup.index_to_string_table_from_file(
        tft_output.vocabulary_file_by_name(constants.ITEM_VOCAB_NAME))
    item_embedding = _get_embedding_matrix(embedding_size, tft_output,
                                           constants.ITEM_VOCAB_NAME)
    item_sims, item_top_k = _get_top_k(tft_features,
                                       item_embedding,
                                       constants.TFT_ITEM_KEY,
                                       item_embedding)
    user_embedding = _get_embedding_matrix(embedding_size, tft_output,
                                           constants.USER_VOCAB_NAME)
    user_sims, user_top_k = _get_top_k(tft_features,
                                       user_embedding,
                                       constants.TFT_USER_KEY,
                                       item_embedding)
    predictions = {
        constants.USER_KEY: tf.identity(features[constants.USER_KEY]),
        "user_top_k": table.lookup(tf.cast(user_top_k, tf.int64)),
        "user_sims": user_sims,
        constants.ITEM_KEY: tf.identity(features[constants.ITEM_KEY]),
        "item_top_k": table.lookup(tf.cast(item_top_k, tf.int64)),
        "item_sims": item_sims,
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  # Eval op
  user_embedding = _update_embedding_matrix(
      tft_features[constants.TFT_USER_KEY],
      user_norm,
      embedding_size,
      tft_output,
      constants.USER_VOCAB_NAME)
  item_embedding = _update_embedding_matrix(
      tft_features[constants.TFT_ITEM_KEY],
      item_norm,
      embedding_size,
      tft_output,
      constants.ITEM_VOCAB_NAME)
  item_sims = tf.matmul(user_norm, item_embedding, transpose_b=True)
  metrics = {}
  with tf.name_scope("recall"):
    for k in constants.EVAL_RECALLS:
      key = "recall_{0}".format(k)
      recall = tf.metrics.recall_at_k(
          tft_features[constants.TFT_TOP_10_KEY], item_sims, k)
      metrics["recall/{0}".format(key)] = recall
      tf.summary.scalar(key, recall[1])
  tf.summary.merge_all()

  preds = tf.reduce_sum(tf.multiply(user_norm, item_norm), axis=1)
  with tf.control_dependencies([user_embedding, item_embedding]):
    loss = tf.losses.mean_squared_error(labels, preds)

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

  # Training op
  optimizer = tf.train.AdagradOptimizer(learning_rate=hparams.learning_rate)
  train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
  return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def _get_trial_id():
  """Returns the trial id if it exists, else "0"."""
  trial_id = json.loads(
      os.environ.get("TF_CONFIG", "{}")).get("task", {}).get("trial", "")
  return trial_id if trial_id else "1"


def get_recommender(params):
  """Returns the product recommendation model."""
  config = tf.estimator.RunConfig(
      save_checkpoints_steps=params.save_checkpoints_steps,
      keep_checkpoint_max=params.keep_checkpoint_max,
      log_step_count_steps=params.log_step_count_steps)
  trial_id = _get_trial_id()
  model_dir = os.path.join(params.model_dir, "trials", trial_id)
  hparams = tf.contrib.training.HParams(
      user_embed_mult=params.user_embed_mult,
      item_embed_mult=params.item_embed_mult,
      learning_rate=params.learning_rate,
      num_layers=params.num_layers)

  model_params = {
      "tft_dir": params.tft_dir,
      "hparams": hparams,
  }
  estimator = tf.estimator.Estimator(
      model_fn=_model_fn,
      model_dir=model_dir,
      config=config,
      params=model_params)
  return estimator

