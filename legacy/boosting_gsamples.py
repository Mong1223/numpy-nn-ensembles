import numpy as np
import random

def get_samples(x, y, prob, shp, num_k=1, num_in_sample=500):
    samples_x = []
    samples_y = []
    samples_prob = []
    samples_ids = []
    for i in range(num_k):
        sample_x = np.zeros(shape=(num_in_sample, 13))
        sample_y = np.zeros(shape=(num_in_sample, 1))
        sample_prob = np.zeros(shape=(num_in_sample, 1))
        sample_ids = np.zeros(shape=(num_in_sample, 1))
        for j in range(num_in_sample):
            if j == num_in_sample:
                pass
            # random_index = random.randrange(int(shp))
            random_index = random.choices(range(shp), weights=prob)
            sample_x[j] = x[random_index]
            sample_y[j] = y[random_index]
            sample_prob[j] = prob[random_index]
            sample_ids[j] = random_index
        # sample_x=np.ndarray(sample_x)
        # sample_y=np.ndarray(sample_y)
        samples_x.append(sample_x)
        samples_y.append(sample_y)
        samples_prob.append(sample_prob)
        samples_ids.append(sample_ids)
    return samples_x, samples_y, samples_prob, samples_ids
