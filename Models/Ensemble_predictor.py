import os
import numpy as np
import tensorflow as tf

class model:
    def __init__(self, path):
        # Model with the conv1d + scaler
        self.model1 = tf.keras.models.load_model(os.path.join(path, 'Conv1D_model'))
        self.mean = np.array([8.4409552, 11.1070953, 21.23054996, 27.24076344, 30.41101099, 26.90711497])
        self.std = np.array([253.58240631, 777.2715952, 722.72048578, 700.70644352, 679.74291031, 712.47014922])

        # Model with BiLSTM + SW
        self.models = {}
        for i in range(7):
            self.models[i] = tf.keras.models.load_model(os.path.join(path, 'SW_Models/'+str(i)))

    def predict(self, X):
        X_s = self.create_sliding(X, 30)
        preds = {}
        for i in range(7):
            preds[i] = self.models[i].predict(X_s[:,:,i,:])

        out2 = np.sum(np.asarray(list(preds.values())), axis=0) / 7

        X = X.numpy()
        for i in range(X.shape[0]):
            X[i, :, :6] = (X[i, :, :6] - self.mean) / self.std

        out1 = self.model1.predict(X)

        out = (out1 + out2) / 2
        out = tf.argmax(out, axis=-1)

        return out

    def create_sliding(self, data, window_size):
        out = np.lib.stride_tricks.sliding_window_view(data, window_size, axis=1)
        out = np.moveaxis(out, 3, 1)
        return out