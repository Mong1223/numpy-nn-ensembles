from nnClassB import my_NN
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from gsamples import get_samples
import random
random.seed(10)
if __name__ == '__main__':
    df = pd.read_csv('heart.csv', header=None)
    df = df.drop(0)
    df[13] = df[13].astype(np.int64)

    # separate the independent variables from the target variable
    data = df.loc[:, 0:12]
    target = df[[13]]

    data = StandardScaler().fit_transform(data)

    # divide the data into a training set and a test set

    x_train, x_test, y_train, y_test = train_test_split(data, target[13], test_size=0.3, random_state=42)

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]).astype('float32').T
    y_train = np.array(y_train).reshape(np.array(y_train).shape[0], 1)
    y_test = np.array(y_test).reshape(np.array(y_test).shape[0], 1).T

    num_k = 5
    shp = x_train.shape[0]
    num_in_sample = x_train.shape[0]
    prob = np.ones(x_train.shape[0]) * 1/x_train.shape[0]

    i_AM = []
    i_pred = []

    for i in range(num_k):
        print('Results ', i+1)
        if i == 0:
            x_traink, y_traink, probk, samples_IDsk = x_train, y_train, prob, np.array(range(x_train.shape[0]))

            clr = my_NN()
            df_X = pd.DataFrame(x_traink.T)  # X sample
            df_Y = pd.DataFrame(y_traink.T)  # Y sample
            x_trainf = df_X.values
            y_trainf = df_Y.values
            df_prob = probk  # Probabilities of example
            df_origin_IDS = samples_IDsk  # Original ids in train data
            clr.train(x_trainf, y_trainf)
            pred = clr.predict(x_train.T)
            AM, new_prob = clr.score(pred, y_trainf, df_prob)
            sc = clr.score_old(pred, y_train.T)
            print('Train_score',sc)
            i_AM.append(AM)
            pred = clr.predict(x_test)
            score = clr.score_old(pred, y_test)
            print('Test score', score)
            i_pred.append(pred)
            print(i, 'AM', AM)
            for l in range(num_in_sample):
                index = int(df_origin_IDS[l])
                prob[index] = new_prob[l]
        else:

            x_traink, y_traink, probk, samples_IDsk = get_samples(x_train, y_train, prob, shp, 1, num_in_sample)

            clr = my_NN()
            df_X = pd.DataFrame(x_traink[0].T) # X sample
            df_Y = pd.DataFrame(y_traink[0].T) # Y sample
            x_trainf = df_X.values
            y_trainf = df_Y.values
            df_prob = probk[0]                 # Probabilities of example
            df_origin_IDS = samples_IDsk[0]    # Original ids in train data
            clr.train(x_trainf, y_trainf)
            pred = clr.predict(x_train.T)
            AM, new_prob = clr.score(pred, y_trainf, df_prob)
            sc = clr.score_old(pred,y_train.T)
            print('Train_score',sc)
            i_AM.append(AM)
            pred = clr.predict(x_test)
            score = clr.score_old(pred,y_test)
            print('test score', score)
            i_pred.append(pred)
            print(i, 'AM', AM)
            for l in range(num_in_sample):
                index = int(df_origin_IDS[l][0])
                prob[index] = new_prob[l]
        print('----------------------')
    print(i_AM)
    # # взвешенное голосование
    # ree = 0
    # for i in range(num_k):
    #     ree += i_AM[i]*i_pred[i]
    #
    # for i in range(len(ree)):
    #     if ree[i]>0.47:
    #         ree[i] = 1
    #     else:
    #         ree[i] = 0
    # print('ree', ree)
    #
    # clr = my_NN()
    # print('reee', clr.score_old(ree,y_test))

    # голосование
    ree = 0
    result = []
    for i in range(num_k):
        ree += i_pred[i]
    for i in ree:
        if i >= (num_k/2):
            result.append(1)
        else:
            result.append(0)
    print('ree', result)
    clr = my_NN()
    print('reee', clr.score_old(result,y_test))

    # print(ree)
        # print('-----------------------')
        # print(prob)
        # print('-----------------------')
        # print(new_prob)








