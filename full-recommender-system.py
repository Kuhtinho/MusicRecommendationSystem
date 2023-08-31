import pandas as pd

from collaborative.matrix_factorization import recommend_songs_matrix_factorization
from diveristy.extreme_feature_track import find_extreme_track_in_playlists
from content_based.k_means import recommend_songs_k_means

data = pd.read_csv("data.csv")
data['artists'] = data['artists'].astype(str)
data['track'] = data.apply(lambda x: x['artists'] + " - " + x['name'], axis=1)

user1 = [
    {"artists": "thirty seconds to mars", "name": "this is war"},
    {"artists": "muse", "name": "resistance"},
    {"artists": "linkin park", "name": "numb"},
    {"artists": "the killers", "name": "mr. brightside"},
    {"artists": "system of a down", "name": "lonely day"}
]


def print_diver_recommendations(tracks):
    k_means_recommendations = recommend_songs_k_means(tracks)
    extreme_track_id = find_extreme_track_in_playlists(tracks)
    matrix_factorization_recommendations = recommend_songs_matrix_factorization(extreme_track_id)

    all_recs = k_means_recommendations + matrix_factorization_recommendations

    for rec in all_recs:
        print(data[data.id == rec].track.values[0])


print_diver_recommendations(user1)