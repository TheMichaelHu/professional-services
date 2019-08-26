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
"""BigQuery queries to feed into Dataflow."""

query = """
WITH 
  partitions AS (
    SELECT
      APPROX_QUANTILES(DATETIME(date_gmt, PARSE_TIME("%R", time_gmt)), 
                       10)[OFFSET(8)] as train_thresh,
      APPROX_QUANTILES(DATETIME(date_gmt, PARSE_TIME("%R", time_gmt)),
                       10)[OFFSET(9)] as validation_thresh
    FROM `bigquery-public-data.epa_historical_air_quality.co_hourly_summary`
    WHERE state_code = "06" AND county_code = "037" AND site_num = "1301"
  ), unwindowed_features AS (
    SELECT sample_measurement, EXTRACT(DAYOFWEEK FROM date_gmt) AS day_of_week,
      EXTRACT(MONTH FROM date_gmt) AS month,
      EXTRACT(DAYOFYEAR FROM date_gmt) AS day_of_year,
      EXTRACT(HOUR FROM PARSE_TIMESTAMP("%R", time_gmt)) AS hour,
      DATETIME(date_gmt, PARSE_TIME("%R", time_gmt)) AS time_stamp,
      CASE 
        WHEN DATETIME(date_gmt, PARSE_TIME("%R", time_gmt)) < train_thresh THEN 0
        WHEN DATETIME(date_gmt, PARSE_TIME("%R", time_gmt)) < validation_thresh THEN 1 
        ELSE 2
      END as part
    FROM `bigquery-public-data.epa_historical_air_quality.co_hourly_summary`, partitions
    WHERE state_code = "06" AND county_code = "037" AND site_num = "1301"
  )
  SELECT ARRAY_AGG(sample_measurement) OVER sliding_window as sample_measurement,
    ARRAY_AGG(day_of_week) OVER sliding_window as day_of_week,
    ARRAY_AGG(month) OVER sliding_window as month,
    ARRAY_AGG(day_of_year) OVER sliding_window as day_of_year,
    ARRAY_AGG(hour) OVER sliding_window as hour,
    MAX(part) OVER sliding_window as part
  FROM unwindowed_features
  WINDOW sliding_window AS (ORDER BY time_stamp ASC ROWS 10 PRECEDING)
"""
