import numpy as np
from assnat.models_base import fit_predict, tokenize_X
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D


def embedding_lstm( X_train, X_test, y_train, y_test):
    X_train, X_test = tokenize_X(X_train, X_test, max_words=5000, pad_max_len=200)

    model = Sequential()
    #Embedding.input_dim	Size of the vocabulary, i.e. maximum integer index + 1.
    #Embedding.output_dim	Dimension of the dense embedding.
    model.add(Embedding(input_dim=X_train.shape[1], output_dim=256))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    return model,  X_train, X_test, y_train, y_test


fit_predict(embedding_lstm, 'leg15', 5, 10, 64)
