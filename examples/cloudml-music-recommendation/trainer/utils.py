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
"""Utility functions for model training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from google.cloud import storage

from constants import constants  # pylint: disable=g-bad-import-order


def _write_projector_metadata_local(model_dir):
  """Write local metadata file to use in tensorboard to visualize embeddings."""
  metadata = os.path.join(model_dir, constants.PROJECTOR_PATH)
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  with open(metadata, "w+") as f:
    f.write("{}\n".format("user") * constants.PROJECTOR_USER_SAMPLES)
    f.write("{}\n".format("item") * constants.PROJECTOR_ITEM_SAMPLES)


def _write_projector_metadata_gcs(model_dir):
  """Write GCS metadata file to use in tensorboard to visualize embeddings."""
  metadata = os.path.join(model_dir, constants.PROJECTOR_PATH)
  split_path = metadata.split("/")
  bucket_name = split_path[2]
  bucket_path = "/".join(split_path[3:])
  storage_client = storage.Client()
  bucket = storage_client.get_bucket(bucket_name)
  blob = bucket.blob(bucket_path)

  _write_projector_metadata_local(".")
  blob.upload_from_filename(constants.PROJECTOR_PATH)
  os.remove(constants.PROJECTOR_PATH)


def write_projector_metadata(model_dir):
  """Write a metadata file to use in tensorboard to visualize embeddings.

  Tensorboard expects a .tsv (tab-seperated values) file encoding information
  about each sample. A header is required if there is more than one column.

  Args:
    model_dir: the directory where the projector config protobuf is written.
  """
  gcs_drive = "gs://"
  if model_dir[:len(gcs_drive)] == gcs_drive:
    _write_projector_metadata_gcs(model_dir)
  else:
    _write_projector_metadata_local(model_dir)
