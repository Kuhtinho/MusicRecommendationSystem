from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import authorization
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

scaler = StandardScaler()

data = pd.read_csv("data.csv")
data = data.drop_duplicates(
  subset = ['artists', 'name'],
  keep = 'last').reset_index(drop = True)
columns = ["valence", "instrumentalness", "danceability", "acousticness", "speechiness", "liveness", "tempo"]

sp = authorization.authorize()


def find_song(track):
    track = track["artists"] + " " + track["name"]
    track = sp.search(track, types=("track",))
    if track is None:
        print('Warning: {} does not exist in Spotify or in database'.format(track['name']))
    return track[0].items[0].id


def calculate_track_vector(track_id):
    track_features = sp.track_audio_features(track_id)
    return np.array([track_features.valence, track_features.instrumentalness,
                     track_features.danceability, track_features.acousticness,
                     track_features.speechiness, track_features.liveness,
                     track_features.tempo])


def get_mean_vector(tracks):
    track_vectors = []

    for track in tracks:
        track_id = find_song(track)
        track_vector = calculate_track_vector(track_id)
        track_vectors.append(track_vector)

    track_matrix = np.array(list(track_vectors))
    return np.mean(track_matrix, axis=0)


track_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                   ('kmeans', KMeans(init="random",
                                                     n_clusters=5,
                                                     n_init=10,
                                                     max_iter=300))
                                   ], verbose=False)
X = data.loc[:, columns]

track_cluster_pipeline.fit(X.values)
song_cluster_labels = track_cluster_pipeline.predict(X.values)


def recommend_songs_k_means(tracks, dataset=data, n_songs=5):
    track_center = get_mean_vector(tracks)
    my_scaler = track_cluster_pipeline.steps[0][1]
    scaled_data = my_scaler.transform(dataset[columns])
    scaled_track_center = my_scaler.transform(track_center.reshape(1, -1))
    distances = cdist(scaled_track_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_tracks = dataset.iloc[index]
    if len(tracks) == 1:
        rec_tracks = rec_tracks[rec_tracks['id'] != find_song(tracks[0])]

    return list(rec_tracks.id.values)

