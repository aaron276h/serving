import data_loader
import numpy as np
from sklearn.preprocessing import StandardScaler


def instance_id(instance_type):
    if instance_type == 'c4.2xlarge':
        return 0
    elif instance_type == 'c4.xlarge':
        return 1
    elif instance_type == 'c4.large':
        return 2
    else:
        assert 0


class DataFetcher:
    def __init__(self, filepaths, is_train, scaler, unify_deltas, unify_instances):
        """

        :param filepaths: List of training files to load
        :param is_train: Specify if this if for training, then the scaler is fit for this data
        :param scaler: pass in the scaler if this is for testing data
        :param unify_deltas: specify delta as a feature
        :param unify_instances: specify instance type as a feature
        :return:
        """
        self.X = []
        self.Y_regression = []
        self.Y_classify = []

        for count, filepath in enumerate(filepaths):
            X, Y_regression, Y_classify = data_loader.load_input_file_modified(input_file_name=filepath)

            # Append with column of deltas
            if unify_deltas:
                delta = float(filepath.split('_')[-1].split('.csv')[0])
                assert 0 < delta < 1
                column_of_deltas = np.ones(shape=[len(X),1]) * delta
                X = np.concatenate([X,column_of_deltas],axis=1)

            if unify_instances:
                instance_type = filepath.split('_')[-2]
                id = instance_id(instance_type=instance_type)
                column_of_ids = np.ones(shape=[len(X),1]) * id
                X = np.concatenate([X,column_of_ids],axis=1)

            if count == 0:
                self.X = X
                self.Y_regression = Y_regression
                self.Y_classify = Y_classify
            else:
                self.X = np.append(self.X, X, axis=0)
                self.Y_regression = np.append(self.Y_regression, Y_regression, axis=0)
                self.Y_classify = np.append(self.Y_classify, Y_classify, axis=0)

        # Create a classification matrix (evicted_true, evicted_false)
        self.Y = np.zeros(shape=[self.Y_classify.size,2])
        for i in range(self.get_size()):
            if self.Y_classify[i] == 0:
                self.Y[i][1] = 1
            else:
                self.Y[i][0] = 1

        # Normalize data with mean 0 and std 1
        self.scaler= None
        if is_train is True:
            scaler = StandardScaler()
            scaler.fit(self.X)
            self.X = scaler.transform(self.X)
            self.scaler = scaler
        else:
            self.scaler = scaler
            self.X = scaler.transform(self.X)

    def get_scaler(self):
        return self.scaler

    def iterate(self, iteration):
        if iteration % 10 == 0:
            perm = np.arange(self.get_size())
            np.random.shuffle(perm)
            self.X = self.X[perm]
            self.Y = self.Y[perm]

    def get_size(self):
        return len(self.X)

    def get_number_of_features(self):
        return self.X.shape[1]

    def fetch_batch_classified(self, batch_number, batch_size):
        offset = batch_number * batch_size
        return self.X[offset:offset+batch_size], self.Y[offset:offset+batch_size]

    def fetch_all_classified(self):
        """ Returns scaled features and classification matrix.
        """
        return self.X, self.Y

    def fetch_all_regression(self):
        """ Returns scaled fatures and time to eviction.
        """
        return self.X, self.Y_regression

    def fetch_one(self, index):
        """ Fetch one sample at a time.
        """
        assert 0 <= index < len(self.X), index
        return self.X[index].reshape(1,self.X.shape[1])