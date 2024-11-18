import tensorflow as tf

class FM_Layer(tf.keras.layers.Layer):
    def __init__(self, k, w_reg, v_reg):
        super().__init__()
        self.k=k # 隐向量维度
        self.w_reg=w_reg # w的正则项系数
        self.v_reg=v_reg # 隐向量矩阵v的正则项系数


    def build(self,input_shape):
        '''
        build初始化可训练参数
        :param input_shape: 输入的shape，输入为n个样本*m个特征矩阵
        :return:
        '''
        self.w0=self.add_weight(name='w0',
                                shape=(1,),
                                initializer=tf.keras.initializers.Zeros(), # 0初始化
                                trainable=True)
        self.w=self.add_weight(name='w',
                               shape=(input_shape[-1],1), # (m,1)，因为需要一个二维矩阵的形式
                               initializer=tf.keras.initializers.RandomNormal(), # 正态分布初始化
                               regularizer=tf.keras.regularizers.l2(self.w_reg), # l2正则化
                               trainable=True)
        self.v=self.add_weight(name='v',
                               shape=(input_shape[-1],self.k), # m*k维的隐向量矩阵
                               initializer=tf.keras.initializers.RandomNormal(),
                               regularizer=tf.keras.regularizers.l2(self.v_reg),
                               trainable=True)
    def call(self,inputs,**kwargs):
        '''
        FM计算逻辑
        :param inputs:n*m矩阵
        :param kwargs:
        :return:output
        '''
        linear_part = tf.matmul(inputs, self.w) + self.w0  # shape:(batchsize, 1)
        inter_part1 = tf.pow(tf.matmul(inputs, self.v), 2)  # shape:(batchsize, self.k)
        inter_part2 = tf.matmul(tf.pow(inputs, 2), tf.pow(self.v, 2))  # shape:(batchsize, self.k)
        inter_part = 0.5 * tf.reduce_sum(inter_part1 - inter_part2, axis=-1, keepdims=True)  # shape:(batchsize, 1)

        output = linear_part + inter_part
        return tf.nn.sigmoid(output)

class FM(tf.keras.Model):
    def __init__(self,k,w_reg=1e-4,v_reg=1e-4):
        super().__init__()
        self.FM_layer=FM_Layer(k=k,w_reg=w_reg,v_reg=v_reg)

    def call(self,inputs,**kwargs):
        output=self.FM_layer(inputs,**kwargs)
        return output


