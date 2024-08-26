import numpy as np
from assnat.models_base import fit_predict
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D


def first_model(X_train, y_train):
    model = Sequential()
    #Embedding.input_dim	Size of the vocabulary, i.e. maximum integer index + 1.
    #Embedding.output_dim	Dimension of the dense embedding.
    model.add(Embedding(input_dim=X_train.shape[1], output_dim=256))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    return model

fit_predict(first_model, 'leg15', 5, 10, 64)
