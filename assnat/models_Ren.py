import numpy as np
import tensorflow as tf
from assnat.models_base import fit_predict, sample, tokenize_X
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from transformers import CamembertTokenizer, TFCamembertModel

def mini_embedding_lstm( X_train, X_test, y_train, y_test):
    X_train, X_test, y_train, y_test=sample(1000, X_train, X_test, y_train, y_test)

    X_train, X_test, vocab_size = tokenize_X(X_train, X_test, max_words=100)

    model = Sequential()
    #Embedding.input_dim	Size of the vocabulary, i.e. maximum integer index + 1.
    #Embedding.output_dim	Dimension of the dense embedding.
    model.add(Embedding(input_dim=vocab_size+1, output_dim=10))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(10, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    return model,  X_train, X_test, y_train, y_test

def embedding_lstm( X_train, X_test, y_train, y_test):
    X_train, X_test, vocab_size = tokenize_X(X_train, X_test, max_words=100000)

    model = Sequential()
    #Embedding.input_dim	Size of the vocabulary, i.e. maximum integer index + 1.
    #Embedding.output_dim	Dimension of the dense embedding.
    model.add(Embedding(input_dim=vocab_size+1, output_dim=100))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    return model,  X_train, X_test, y_train, y_test

from transformers import TFCamembertModel, CamembertTokenizer
import tensorflow as tf

def camembert_tokenize_X(X_train, X_test):
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

    def tokenize(texts):
        tokens = tokenizer(texts.tolist(), padding=True, truncation=True, return_tensors="np")
        return {"input_ids": tokens['input_ids'], "attention_mask": tokens['attention_mask']}

    X_train_tokenized = tokenize(X_train)
    X_test_tokenized = tokenize(X_test)

    return X_train_tokenized, X_test_tokenized

def camembert(X_train, X_test, y_train, y_test):
    #X_train, X_test, y_train, y_test=sample(1000, X_train, X_test, y_train, y_test)
    # Tokeniser les données d'entrée
    X_train, X_test = camembert_tokenize_X(X_train, X_test)

    # Préparer les données pour être directement compatibles avec model.fit
    X_train_prepared = (X_train['input_ids'], X_train['attention_mask'])
    X_test_prepared = (X_test['input_ids'], X_test['attention_mask'])

    # Définir les entrées du modèle
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='attention_mask')

    # Charger CamemBERT et l'utiliser pour les embeddings
    camembert_model = TFCamembertModel.from_pretrained('camembert-base')
    camembert_output = camembert_model([input_ids, attention_mask]).last_hidden_state[:, 0, :]

    # Ajouter une couche Dense pour la classification finale
    output_layer = tf.keras.layers.Dense(7, activation='softmax')(camembert_output)  # Sigmoid pour une classification binaire

    # Créer le modèle
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output_layer)

    # Compiler le modèle
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Retourner le modèle ainsi que les données préparées pour l'entraînement
    return model, X_train_prepared, X_test_prepared, y_train, y_test


#fit_predict(mini_embedding_lstm, 'leg15', 5, 10, 64)
fit_predict(camembert, 'leg15', 5, 1, 64)
