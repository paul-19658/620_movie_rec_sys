import numpy as np
import data_loader

path1='../Data'
file_movies='movies.csv'
DL=data_loader.DataLoader(path1)


data_movies=DL.load_csv(file_name=file_movies)
print(data_movies.head())
DL.one_hot_encode()


# DL.check_missing_values(data=data)