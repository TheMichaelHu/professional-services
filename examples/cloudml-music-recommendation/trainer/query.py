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
"""BigQuery query to feed into Dataflow."""

query = """
  WITH
    user_songs as (
      SELECT user_name as user, ANY_VALUE(artist_name) as artist,
        CONCAT(track_name, " by ", artist_name) as song,
        COUNT(*) as user_song_listens
      FROM `listenbrainz.listenbrainz.listen`
      WHERE track_name != ""
      GROUP BY user_name, song
      HAVING user_song_listens > 1
    ),
    user_song_ranks as (
      SELECT user, song, user_song_listens,
        ROW_NUMBER() OVER (PARTITION BY user ORDER BY user_song_listens DESC)
          AS rank
      FROM user_songs
    ),
    user_features as (
      SELECT user, ARRAY_AGG(song) as top_10,
        MAX(user_song_listens) as user_max_listen
      FROM user_song_ranks
      WHERE rank <= 10
      GROUP BY user
    ),
    item_features as (
      SELECT CONCAT(track_name, " by ", artist_name) as song,
        SPLIT(ANY_VALUE(tags), ",") as tags,
        COUNT(DISTINCT(track_name)) as albums
      FROM `listenbrainz.listenbrainz.listen`
      WHERE track_name != ""
      GROUP BY song
    )
  SELECT user, song, artist, tags, albums, top_10,
    user_song_listens/user_max_listen as count_norm
  FROM user_songs
  JOIN user_features USING(user)
  JOIN item_features USING(song)
"""
