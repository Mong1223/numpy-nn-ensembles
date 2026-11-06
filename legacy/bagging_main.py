import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from nnClass import my_NN
from gsamples import get_samples

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

    num_k = 3
    shp = x_train.shape[0]
    num_in_sample = 700
    features = 13
    results = []
    for k in range(1, num_k + 1, 2):
        num = k
        # Выборки
        x_traink, y_traink = get_samples(x_train, y_train, shp, num, num_in_sample)

        Parameters_f = []

        for i in range(num):
            clr = my_NN()
            df_X = pd.DataFrame(x_traink[i].T)
            df_Y = pd.DataFrame(y_traink[i].T)

            x_trainf = df_X.values
            y_trainf = df_Y.values
            clr.train(x_trainf, y_trainf, iteration=10000)
            pre_y = clr.predict(x_test)  # predict

            Parameters_f.append(pre_y)

        res = 0
        for i in range(num):
            res = res + Parameters_f[i]
        result = []
        for i in res:
            if i >= (num / 2):
                result.append(1)
            else:
                result.append(0)
        clr = my_NN()
        score = clr.score(result, y_test)
        # print('res', result)
        # print('y', Y_test)
        print(score)
        results.append([k, score])

        print('hi')
    print(results)

