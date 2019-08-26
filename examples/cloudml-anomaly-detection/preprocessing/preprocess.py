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
"""Builds preprocessing pipeline for product recommendation, producing user-item pairs."""

import math
import os

import apache_beam as beam
import numpy as np
from tensorflow_transform import coders
from tensorflow_transform.tf_metadata import dataset_schema

# pylint: disable=g-bad-import-order
from constants import constants
from preprocessing import query as bq_query


# pylint: disable=invalid-name
# pylint: disable=expression-not-assigned
# pylint: disable=no-value-for-parameter
@beam.ptransform_fn
def ReadBQ(p, query):
  """Ingests BQ query results into the pipeline.

  Args:
    p: a pCollection to read the data into.
    query: a BigQuery query.

  Returns:
    A pCollection of injested inputs.
  """
  return p | "ReadQuery" >> beam.io.Read(beam.io.BigQuerySource(
      query=query, use_standard_sql=True))


def _build_features(data):
  """."""
  features = {}
  for cycle in constants.CYCLIC_DATA:
    feature, period = cycle["feature"], cycle["period"]
    features[feature + "_sin"] = [math.sin(2 * math.pi * x / period)
                                  for x in data[feature]]
    features[feature + "_cos"] = [math.cos(2 * math.pi * x / period)
                                  for x in data[feature]]
  return (data[constants.PARTITION_KEY], features)


def _split_data(p):
  """Splits the data into train/validation/test."""
  split = p | "SplitData" >> beam.Partition(
      lambda x, _: int(x[0]), 3)
  return zip([constants.TRAIN, constants.VAL, constants.TEST], split)


@beam.ptransform_fn
def Shuffle(p):
  """Shuffles the given pCollection."""
  return (p
          | "PairWithRandom" >> beam.Map(lambda x: (np.random.random(), x))
          | "GroupByRandom" >> beam.GroupByKey()
          | "DropRandom" >> beam.FlatMap(lambda x: x[1]))


@beam.ptransform_fn
def WriteOutput(p, prefix, output_dir, feature_spec, plain_text=False):
  """Writes the given pCollection as a TF-Record.

  Args:
    p: a pCollection.
    prefix: prefix for location tf-record will be written to.
    output_dir: the directory or bucket to write the json data.
    feature_spec: the feature spec of the tf-record to be written.
    plain_text: if true, write the output as plain text instead.
  """
  path = os.path.join(output_dir, prefix)
  shuffled = p | "ShuffleData" >> Shuffle()

  if plain_text:
    shuffled | "WriteToText" >> beam.io.WriteToText(
        path, file_name_suffix=".txt")
    return

  schema = dataset_schema.from_feature_spec(feature_spec)
  coder = coders.ExampleProtoCoder(schema)
  shuffled | "WriteTFRecord" >> beam.io.tfrecordio.WriteToTFRecord(
      path,
      coder=coder,
      file_name_suffix=".tfrecord")


def run(p, args):
  """Creates a pipeline to build and write train/val/test datasets."""
  query = bq_query.query
  if not args.cloud:
    query = "{} LIMIT {}".format(query, 10)

  data = (p
          | "ReadBQ" >> ReadBQ(query)
          | "BuildFeatures" >> beam.Map(_build_features))

  data = _split_data(data)
  for name, dataset in data:
    (dataset
     | "DropParitionKey{}".format(name) >> beam.Map(lambda x: x[1])
     | "Write{}Output".format(name) >> WriteOutput(
         name, args.output_dir, constants.TRAIN_SPEC, args.plain_text))
