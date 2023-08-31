import pandas as pd
import numpy as np
import authorization

sp = authorization.authorize()
columns = ["valence", "instrumentalness", "danceability", "acousticness", "speechiness", "liveness", "tempo"]

df = pd.read_csv('merged_data.csv')

size = lambda x: len(x)
df_freq = df.groupby(['id', 'user_id']).agg('size').reset_index().rename(columns={0:'freq'})[['id','user_id', 'freq']].sort_values(['freq'], ascending=False)


def mapper(col):
    coded_dict = dict()
    cter = 1
    encoded = []

    for val in df_freq[col]:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter += 1

        encoded.append(coded_dict[val])
    return encoded


track_ids = mapper('id')
user_ids = mapper('user_id')

df_freq['track_ids'] = track_ids
df_freq['user_ids'] = user_ids
df_freq.head()


ratings_mat = np.ndarray(
    shape=(np.max(df_freq.track_ids.values), np.max(df_freq.user_ids.values)),
    dtype=np.uint8)
ratings_mat[df_freq.track_ids.values-1, df_freq.user_ids.values-1] = df_freq.freq.values

normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)


def top_cosine_similarity(data, sliced_data, track_id, top_n=5):
    index = int(data[data['id'] == track_id].index[0])
    track_row = sliced_data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', sliced_data, sliced_data))
    similarity = np.dot(track_row, sliced_data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]


track_data = df.drop_duplicates(
  subset = ['artists', 'name'],
  keep = 'last').reset_index(drop = True)
track_data = df.drop_duplicates(subset = ['id'], keep = 'last').reset_index(drop = True)
track_data['track'] = track_data.apply(lambda x: x['artists'] + " - " + x['name'], axis=1)


def print_similar_tracks(track_data, track_id, top_indexes):
    print('Recommendations for {0}: \n'.format(
    track_data[track_data.id == track_id].track.values[0]))
    for id in top_indexes+1:
        track_id = track_data.iloc[id]['id']
        print(track_data[track_data.id == track_id].track.values[0])


k = 50
top_n = 5
sliced = V.T[:, :k]


def recommend_songs_matrix_factorization(track_id):
    indexes = top_cosine_similarity(track_data, sliced, track_id, top_n)
    track_ids = []

    for id in indexes + 1:
        track_id = track_data.iloc[id]['id']
        track_ids.append(track_id)

    return track_ids