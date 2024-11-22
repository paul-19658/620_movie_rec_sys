import os
from typing import Tuple, Any

import pandas as pd
import logging

from pandas import DataFrame, Series

# 初始化日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, raw_data_path: str, processed_data_path: str = None):
        """
        初始化数据加载器
        :param raw_data_path: 原始数据存储路径
        :param processed_data_path: 处理后的数据存储路径（可选）
        """
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

    def load_csv(self, file_name: str) -> pd.DataFrame:
        """
        加载 CSV 文件为 DataFrame。
        :param file_name: 文件名
        :return: pandas DataFrame
        """
        file_path = os.path.join(self.raw_data_path, file_name)
        if not os.path.exists(file_path):
            logger.error(f"文件未找到：{file_path}")
            raise FileNotFoundError(f"文件未找到：{file_path}")

        try:
            logger.info(f"加载文件：{file_path}")
            data = pd.read_csv(file_path)
            logger.info(f"文件加载成功，形状为：{data.shape}")
            return data
        except Exception as e:
            logger.error(f"加载文件时出错：{e}")
            raise

    def check_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        检查数据中的缺失值。
        :param data: pandas DataFrame
        :return: 缺失值统计表
        """
        missing = data.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            logger.info("数据中没有缺失值。")
            return pd.DataFrame()
        else:
            logger.warning(f"缺失值统计：\n{missing}")
            return missing.to_frame(name='missing_count')

    def one_hot_encode(self, data_movies: pd.DataFrame, data_ratings: pd.DataFrame) -> tuple[DataFrame, Any]:
        # 先合并
        data = pd.merge(data_ratings, data_movies, on='movieId', how='inner')

        # 提取年份信息并进行分桶
        data['year'] = data['title'].str.extract(r'\((\d{4})\)')
            # 先转为int
        data['year'] = data['year'].astype('float')
            # 定义分桶范围（边界需要包含最大值 2020）
        bins = list(range(1900, 2030, 10))  # 每隔 10 年一个分桶
        labels = [f"{start}-{end - 1}" for start, end in zip(bins[:-1], bins[1:])]
            # 使用 pd.cut 进行分桶
        year_bucket0 = pd.cut(data['year'], bins=bins, labels=labels, right=False)
        year_bucket1= pd.get_dummies(year_bucket0,prefix='year_bucket').astype('int')

        # 拆分genres列，并进行one-hot编码
        genres_split = data['genres'].str.get_dummies(sep='|')

        # 对movieId进行one-hot处理
        # newMovieId = pd.get_dummies(data['movieId'], prefix='movieId')
        # newMovieId = newMovieId.astype('int')
        # print(newId.head(10))

        #对userId进行one-hot处理
        newUserId = pd.get_dummies(data['userId'], prefix='userId')
        newUserId = newUserId.astype('int')

        # 合并处理好的数据
        data = pd.concat([data, genres_split, newUserId,year_bucket1], axis=1)

        # 删除原来的title和genres列
        data.drop(columns=['title', 'genres', 'userId','year','movieId'], inplace=True)

        # 将rating作为y，其他列作为X
        y = data['rating']
        X = data.drop(columns=['rating', 'timestamp'])  # 删除rating和无关的timestamp列
        return X, y

