import numpy as np
import data_loader

path1='../Data'
file_movies='movies.csv'
DL=data_loader.DataLoader(path1)


data_movies=DL.load_csv(file_name=file_movies)
print(data_movies.head())
DL.one_hot_encode()


def one_hot_encode(self, data_movies: pd.DataFrame, data_ratings: pd.DataFrame) -> tuple[DataFrame, Any]:
    # 先合并
    data = pd.merge(data_ratings, data_movies, on='movieId', how='inner')

    # 第一部分处理data——movies，提取title中的年份做one-hot，以及为genres做one-hot
    # 提取年份信息并进行one-hot编码
    data['year'] = data['title'].str.extract(r'\((\d{4})\)')
    year_one_hot = pd.get_dummies(data['year'], prefix='year')
    year_one_hot = year_one_hot.astype('int')
    # 拆分genres列，并进行one-hot编码
    genres_split = data['genres'].str.get_dummies(sep='|')

    # 对movieId进行one-hot处理
    newId = pd.get_dummies(data['movieId'], prefix='movieId')
    newId = newId.astype('int')
    # print(newId.head(10))
    ###

    # 合并处理好的数据
    data = pd.concat([data, year_one_hot, genres_split, newId], axis=1)

    # 删除原来的title和genres列
    data.drop(columns=['title', 'genres', 'year'], inplace=True)

    # 第二部分，合并两张表，return X,y
    # 合并电影信息表和评分表
    # data = pd.merge(data_ratings, data_movies, on='movieId', how='inner')

    # 将rating作为y，其他列作为X
    y = data['rating']
    X = data.drop(columns=['rating', 'timestamp'])  # 删除rating和无关的timestamp列
    return X, y

####
    def one_hot_encode(self, data_movies: pd.DataFrame, data_ratings: pd.DataFrame) -> tuple[DataFrame, Any]:
        # 先合并
        data = pd.merge(data_ratings, data_movies, on='movieId', how='inner')

        # 第一部分处理data——movies，提取title中的年份做one-hot，以及为genres做one-hot
        # 提取年份信息并进行one-hot编码
        data_movies['year'] = data_movies['title'].str.extract(r'\((\d{4})\)')
        year_one_hot = pd.get_dummies(data_movies['year'], prefix='year')
        year_one_hot = year_one_hot.astype('int')
        # 拆分genres列，并进行one-hot编码
        genres_split = data_movies['genres'].str.get_dummies(sep='|')

        # 对movieId进行one-hot处理
        newId=pd.get_dummies(data_movies['movieId'], prefix='movieId')
        newId=newId.astype('int')
        # print(newId.head(10))
        ###

        # 合并处理好的数据
        data_movies = pd.concat([data_movies, year_one_hot, genres_split, newId], axis=1)

        # 删除原来的title和genres列
        data_movies.drop(columns=['title', 'genres','year'], inplace=True)

        # 第二部分，合并两张表，return X,y
        # 合并电影信息表和评分表
        # data = pd.merge(data_ratings, data_movies, on='movieId', how='inner')

        # 将rating作为y，其他列作为X
        y = data['rating']
        X = data.drop(columns=['rating', 'timestamp'])  # 删除rating和无关的timestamp列
        return X, y

# DL.check_missing_values(data=data)