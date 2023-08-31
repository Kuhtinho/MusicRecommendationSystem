import pandas as pd
import numpy as np
import authorization
from scipy.spatial import distance
import users_information

data = pd.read_csv("../data.csv")
columns = ["id", "valence", "instrumentalness", "danceability", "acousticness", "speechiness", "liveness", "tempo"]
columns_without_id = ["valence", "instrumentalness", "danceability", "acousticness", "speechiness", "liveness", "tempo"]
data = data[columns]
sp = authorization.authorize()


def calculate_ild_for_songs(songs):
    distances = []
    for i in range(len(songs)):
        for j in range(i+1, len(songs)):
            dist = distance.cosine(songs[i], songs[j])
            distances.append(dist)
    return np.mean(distances)


def calculate_ild(users):
    total_ILD = 0
    tracks = []
    for user, track_ids in users.items():
        for track_id in track_ids:
            track = data.loc[data['id'] == track_id]
            track = track[columns_without_id]
            tracks.append(track.values.tolist()[0])
        total_ILD += calculate_ild_for_songs(tracks)
    return total_ILD / len(users)


user_data = {
    "user1": users_information.user1_recs_diver,
    "user2": users_information.user2_recs_diver,
    "user3": users_information.user3_recs_diver,
    "user4": users_information.user4_recs_diver,
    "user5": users_information.user5_recs_diver,
    "user6": users_information.user6_recs_diver,
    "user7": users_information.user7_recs_diver,
    "user8": users_information.user8_recs_diver,
    "user9": users_information.user9_recs_diver,
    "user10": users_information.user10_recs_diver,
    "lexa": users_information.lexa_recs_diver,
    "makar1": users_information.makar1_recs_diver,
    "makar2": users_information.makar2_recs_diver,
    "ola1": users_information.ola1_recs_diver,
    "ola2": users_information.ola2_recs_diver,
    "tania1": users_information.tania1_recs_diver,
    "tania2": users_information.tania2_recs_diver,
    "sergey": users_information.sergey_recs_diver,
    "basia": users_information.basia_recs_diver,
    "nikita": users_information.nikita_recs_diver
}

user_data_without_diver = {
    "user1": users_information.user1_recs_withour_diver,
    "user2": users_information.user2_recs_without_diver,
    "user3": users_information.user3_recs_without_diver,
    "user4": users_information.user4_recs_without_diver,
    "user5": users_information.user5_recs_without_diver,
    "user6": users_information.user6_recs_without_diver,
    "user7": users_information.user7_recs_without_diver,
    "user8": users_information.user8_recs_without_diver,
    "user9": users_information.user9_recs_without_diver,
    "user10": users_information.user10_recs_without_diver,
    "lexa": users_information.lexa_recs_without_diver,
    "makar1": users_information.makar1_recs_without_diver,
    "makar2": users_information.makar2_recs_without_diver,
    "ola1": users_information.ola1_recs_without_diver,
    "ola2": users_information.ola2_recs_without_diver,
    "tania1": users_information.tania1_recs_without_diver,
    "tania2": users_information.tania2_recs_without_diver,
    "sergey": users_information.sergey_recs_without_diver,
    "basia": users_information.basia_recs_without_diver,
    "nikita": users_information.nikita_recs_without_diver
}

print("With diversity:")
print(str(calculate_ild(user_data)))
print()
print("Without diversity:")
print(str(calculate_ild(user_data_without_diver)))
