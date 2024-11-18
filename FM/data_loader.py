import os
import pandas as pd
import logging

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
    def one_hot_encode(self, data_movies: pd.DataFrame, data_ratings: pd.DataFrame) -> object:

        # 第一部分处理data——movies，提取title中的年份做one-hot，以及为genres做one-hot
        # 提取年份信息并进行one-hot编码
        data_movies['year'] = data_movies['title'].str.extract(r'\((\d{4})\)')
        year_one_hot = pd.get_dummies(data_movies['year'], prefix='year')
        year_one_hot = year_one_hot.astype('int')
        # 拆分genres列，并进行one-hot编码
        genres_split = data_movies['genres'].str.get_dummies(sep='|')

        # 合并处理好的数据
        data_movies = pd.concat([data_movies, year_one_hot, genres_split], axis=1)

        # 删除原来的title和genres列
        data_movies.drop(columns=['title', 'genres','year'], inplace=True)

        # 第二部分，合并两张表，return X,y
        # 合并电影信息表和评分表
        data = pd.merge(data_ratings, data_movies, on='movieId', how='inner')

        # 将rating作为y，其他列作为X
        y = data['rating']
        X = data.drop(columns=['rating', 'timestamp'])  # 删除rating和无关的timestamp列
        return X, y

