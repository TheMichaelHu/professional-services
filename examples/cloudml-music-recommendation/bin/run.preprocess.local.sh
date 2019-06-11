#!/bin/bash

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

# Convenience script for running wals preprocessing pipeline locally.
. ./bin/_common.sh

NOW="$(get_date_time)"
PROJECT_ID="$(get_project_id)"
BUCKET="gs://${PROJECT_ID}-bucket"
TFT_PATH="${OUTPUT_DIR}/tft_${NOW}/"

python -m preprocessing.run_preprocess \
  --output_folder "${OUTPUT_DIR}" \
  --tft_dir "${TFT_PATH}"\
  --user_min_count 0 \
  --item_min_count 0 \
  --plain_text
