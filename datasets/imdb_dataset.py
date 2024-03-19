import os
from pathlib import Path
import pandas as pd

# data link: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset


def create_title_cast_dataset(save_path):
    credit_data_path = './datasets/The-Movies-Dataset/credits.csv'
    credit_data = pd.read_csv(credit_data_path)

    movies_data_path = './datasets/The-Movies-Dataset/movies_metadata.csv'
    movies_data = pd.read_csv(movies_data_path)

    new_data = pd.DataFrame(columns=['id', 'title', 'release_date', 'cast'])

    for idx, credit_row in credit_data.iterrows():
        cast = eval(credit_row['cast'])
        movie_id = credit_row['id']

        movie_row = movies_data.loc[movies_data['id'] == str(movie_id)]
        if len(movie_row) != 1:
            continue

        title = movie_row['title'].item()
        release_date = movie_row['release_date'].item()

        new_data.loc[len(new_data)] = [movie_id, title, release_date, cast]

    new_data.dropna(inplace=True)
    new_data.to_csv(save_path, index=False)


# def create_actors_film_dataset():
#     data_path = '/mnt/nfs/yakir/hallucination_project/datasets/IMDB-Films/actorfilms.csv'
#     data = pd.read_csv(data_path)
#
#     film_ids = data['FilmID'].unique()
#
#     new_data = pd.DataFrame(columns=['id', 'title', 'actors'])
#     for film_id in film_ids:
#         film_data = data.loc[data['FilmID'] == film_id]
#         actors = film_data['Actor'].tolist()
#         title = film_data['Film'].tolist()[0]
#
#         new_data.loc[len(new_data)] = [film_id, title, actors]
#
#     new_data.to_csv('/mnt/nfs/yakir/hallucination_project/datasets/IMDB-Films/film_with_actors.csv', index=False)
#

if __name__ == '__main__':
    dataset_save_path = './datasets/The-Movies-Dataset/title_with_cast.csv'
    if not os.path.isfile(dataset_save_path):
        print('Create title-cast imdb dataset...')
        create_title_cast_dataset(dataset_save_path)
        print('Dataset title-cast created.')
    else:
        print('The dataset title-cast already exists')
