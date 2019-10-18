import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
import matplotlib.pyplot as plt
from keras.layers import Dense, Flatten, Dropout
from keras.layers import LSTM
import pandas as pd
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D
from keras import backend as K
from keras.layers import Lambda,TimeDistributed
#from keras.callbacks import EarlyStopping
from keras_self_attention import SeqSelfAttention


"""
搭建模型
"""     
def create_model():
    print('---Build model---')
    model = Sequential()
    model.add(Conv1D(
                 filters=13,kernel_size=2,
                 activation='relu',
                 input_shape=(x_train.shape[1],1)))
    model.add(Conv1D(filters=26,kernel_size=2,
                 activation='relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(TimeDistributed(Flatten())) 
    model.add(TimeDistributed(Dense(units=128, activation='relu')))
    model.add(LSTM(64,  return_sequences=True))
    
    SeqSelfAttention(
    attention_width=15,
    attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL,
    attention_activation=None,
    use_attention_bias=False,
    name='Attention')
    
    model.add(Lambda(lambda x: K.max(x, axis=1), output_shape=(64,)))
    model.add(Dropout(0.5)) # Dropout overfitting
    model.add(Dense(2, activation='sigmoid', name='dense_1'))
    
    # 编译模型
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy',precision,recall,F1])

    model.summary()#打印出模型的参数
    return model





if __name__ == '__main__':
    
    
    """
    路径
    """
    path =r"...\data.xlsx"
    x_train, x_test, y_train, y_test, y_test_origin, x_train2, x_valid, y_train2, y_valid, batch_size_0 = load_data(path)
    model=create_model()    
    
    history=model.fit(x_train2, y_train2,
          epochs=200,
          batch_size=128,
          validation_data=(x_valid, y_valid))
    
    
    print('中间层结果可视化......')
    intermediate_layer_model_dense_1 = Model(inputs=model.input, 
                                 outputs=model.get_layer('dense_1').output)
    get_feature= intermediate_layer_model_dense_1
    dense_1_train_feature = get_feature.predict(x_train)  #(x_train2)
    dense_1_test_feature = get_feature.predict(x_test)
    np.save('Train_Test_Set/dense_1_train_feature.npy', dense_1_train_feature)
    np.save('Train_Test_Set/dense_1_test_feature.npy', dense_1_test_feature)

    #TrainData = np.load('Train_Test_Set/dense_1_train.npy')
    #TestData = np.load('Train_Test_Set/dense_1_test.npy')

    print('the shape of dense_1_train_feature:',dense_1_train_feature.shape)
    print('the shape of dense_1_test_feature:',dense_1_test_feature.shape)

    print('dense_1_train',dense_1_train_feature[0])
    print('dense_1_test',dense_1_test_feature[0])
    
    
    from sklearn.neighbors import KNeighborsClassifier
    y_train_knn=y_train
    y_test_knn=y_test
    k = 7 
    knc = KNeighborsClassifier(n_neighbors = k)
    knc.fit(dense_1_train_feature,y_train_knn)
    y_predict = knc.predict(dense_1_test_feature)
