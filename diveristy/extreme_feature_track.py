from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import authorization

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

scaler = StandardScaler()
sp = authorization.authorize()

df_tracks = pd.read_csv("../data.csv")
df_tracks_with_playlists = pd.read_csv('../merged_data.csv')


columns = ["valence", "instrumentalness", "danceability", "acousticness", "speechiness", "liveness", "tempo"]


track_cluster_pipeline = Pipeline([('scaler', StandardScaler()),
                                   ('kmeans', KMeans(init="random", n_clusters=5,n_init=10,max_iter=300,random_state=42))
                                   ], verbose=False)
X = df_tracks_with_playlists.loc[:, columns]

track_cluster_pipeline.fit(X.values)
song_cluster_labels = track_cluster_pipeline.predict(X.values)


def find_extreme_track_in_playlists(tracks):
    extreme_track_id = find_extreme_track(tracks)[0]
    extreme_track_data = get_track_vector(extreme_track_id)
    my_scaler = track_cluster_pipeline.steps[0][1]
    scaled_data = my_scaler.transform(df_tracks_with_playlists[columns])
    scaled_track_center = my_scaler.transform((np.array(extreme_track_data)).reshape(1, -1))
    distances = cdist(scaled_track_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :1][0])

    similar_track = df_tracks_with_playlists.iloc[index]

    return similar_track["id"].iloc[0]


def find_extreme_track(tracks):
    track_features_dicts = []
    track_ids_with_dicts = []
    track_vector = None
    track_id = None

    for track in tracks:
        track_id = find_track(track)
        track_features_dict = get_track_feature_dict(track_id)
        track_info = {track_id: track_features_dict}
        track_ids_with_dicts.append(track_info)
        track_features_dicts.append(track_features_dict)

    feature_values_dicts = flatten_dict_list(track_features_dicts)

    feature, value = find_extreme_feature_in_track(feature_values_dicts)

    for track_features_dict in track_features_dicts:
        if track_features_dict[feature] == value:
            track_vector = list(track_features_dict.values())

    for track_id_with_dict in track_ids_with_dicts:
        for key, track_features_dict in track_id_with_dict.items():
            if list(track_features_dict.values()) == track_vector:
                track_id = key

    return track_id, feature, value


def find_extreme_feature_in_track(values_in_dicts):
    max_extreme_val = 0
    extreme_feature_in_track = None
    for key, value in values_in_dicts.items():
        extreme_val = find_extreme_value(value)
        if extreme_val[1] >= max_extreme_val:
            max_extreme_val = extreme_val[1]
            extreme_feature_in_track = key, extreme_val[1]
    return extreme_feature_in_track


def find_extreme_value(values):
    max_value = max(values)
    min_value = min(values)

    second_min = sorted(values)[1]
    second_max = sorted(values, reverse=True)[1]

    if max_value - second_max == second_min - min_value:
        return "min_value", min_value
    elif (max_value - second_max) > (second_min - min_value):
        return "max_value", max_value
    else:
        return "min_value", min_value


def flatten_dict_list(dict_list):
    flattened_dict = {}
    for key in dict_list[0].keys():
        flattened_dict[key] = []

    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)

    return flattened_dict


def get_track_vector(track_id):
    track_features = sp.track_audio_features(track_id)
    return np.array([track_features.valence, track_features.instrumentalness,
                     track_features.danceability, track_features.acousticness,
                     track_features.speechiness, track_features.liveness, track_features.tempo])


def get_track_feature_dict(track_id):
    track_features = sp.track_audio_features(track_id)
    return {"valence": track_features.valence, "instrumentalness": track_features.instrumentalness,
            "danceability": track_features.danceability, "acousticness": track_features.acousticness,
            "speechiness": track_features.speechiness, "liveness": track_features.liveness,
            "tempo": track_features.tempo/250}


def find_track(track):
    track = track["artists"] + " " + track["name"]
    track = sp.search(track, types=("track",))
    if track is None:
        print('Warning: {} does not exist in Spotify or in database'.format(track['name']))
    return track[0].items[0].id

