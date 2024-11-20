from sklearn.model_selection import train_test_split
from model import FM
import tensorflow as tf
import data_loader
from sklearn.metrics import mean_squared_error

# 加载数据
DataPath='../Data'
file_movies='movies.csv'
file_ratings='ratings.csv'
DL=data_loader.DataLoader(DataPath)

data_movies=DL.load_csv(file_name=file_movies)
data_ratings=DL.load_csv(file_name=file_ratings)
# 检测缺失值
DL.check_missing_values(data_movies)
DL.check_missing_values(data_ratings)
# one-hot编码
X,y=DL.one_hot_encode(data_movies=data_movies,data_ratings=data_ratings)

# print(X.head(10))
# print(X.shape)
# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print(X_train.shape)
# print(y_train.shape)
# # 转为float，不然报错
X_train=X_train.astype('float32') #32会提示内存不够
X_test=X_test.astype('float32')

# 初始化模型
k = 10
w_reg=1e-6
v_reg=1e-6
model=FM(k=k,w_reg=w_reg,v_reg=v_reg)
optimizer=tf.keras.optimizers.SGD(learning_rate=0.0003)
model.compile(loss='mse',optimizer=optimizer)

# 训练
model.fit(X_train,y_train,epochs=50)

# 评估
predictions=model.predict(X_test)
print(mean_squared_error(y_test,predictions))