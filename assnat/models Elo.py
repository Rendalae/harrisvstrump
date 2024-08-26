import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import string
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import text_to_word_sequence
from assnat.params import *
from assnat.clean import complete_preproc
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

import os

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
# Include the epoch in the file name (uses `str.format`)



def train_model_RNN(leg_, min_words_, patience_, epoch_, embedding_dim_, max_len_):
    # Chargement et prétraitement des données
    df = pd.read_csv(leg_)
    df_preproc = complete_preproc(df, na_col=["Texte", "famille"], drop_names=["Mme la présidente", "M. le président"], min_words=min_words_, punct_opt=True)

    X = df_preproc['Texte']  # Les textes à classifier
    y = df_preproc['famille']

    # Encodage des labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Tokenization des textes
    max_words = 5000
    max_len = max_len_  # Limiter les séquences à max_len_ tokens pour l'entraînement
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Remplir les séquences pour qu'elles aient toutes la même longueur
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    # Construire le modèle RNN avec LSTM
    embedding_dim = embedding_dim_
    early_stopper = EarlyStopping(monitor='val_loss', patience=patience_, restore_best_weights=True)


    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

    # Compiler le modèle
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Afficher un résumé du modèle
    model.summary()

    # Entraîner le modèle
    batch_size = 64
    epochs = epoch_

    checkpoint_path = "training_2/cp-{epoch:04d}.ckpt" # A modifier
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=5*batch_size)
    # Save the weights using the `checkpoint_path` format
    model.save_weights(checkpoint_path.format(epoch=0))


    history = model.fit(
        X_train_padded, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_padded, y_test),
        callbacks=[early_stopper], [cp_callback]
        verbose=2
    )

    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test_padded)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Afficher le rapport de classification
    print("Rapport de classification pour le RNN avec LSTM:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))


train_model_RNN(leg_='data/leg16.csv', min_words_=10, patience_=3, epoch_=1, embedding_dim_=100, max_len_=100)

'''def model_RNN(leg_=leg_, min_words_=min_words_, patience_=patience_, epoch_=epoch_ , embedding_dim_=embedding_dim_ , max_len_= max_len_)
    df = pd.read_csv('leg_')
    df_preproc = complete_preproc(df, na_col = ["Texte","famille"], drop_names = ["Mme la présidente","M. le président"], min_words = min_words_, punct_opt = True)

    X = df_preproc['Texte']  # Les textes à classifier
    y = df_preproc['famille']

    # Encodage des labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    # Tokenization des textes
    max_words = 5000
    max_len = max_len_token # Limiter les séquences à 100 tokens pour l'entraînement
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    # Remplir les séquences pour qu'elles aient toutes la même longueur
    X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
    X_test_padded = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')

    # Construire le modèle RNN avec LSTM
    embedding_dim = embedding_dim
    early_stopper = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    model = Sequential()
    model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

    # Compiler le modèle
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Afficher un résumé du modèle
    model.summary()

    # Entraîner le modèle
    batch_size = 64
    epochs = epoch_

    history = model.fit(
        X_train_padded, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test_padded, y_test),
        callbacks=[early_stopper],
        verbose=2
    )

    # Prédire sur l'ensemble de test
    y_pred = model.predict(X_test_padded)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Afficher le rapport de classification
    print("Rapport de classification pour le RNN avec LSTM:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))'''

'''def train_model_classicML(input_leg, drop_fam, na_col, drop_names, min_words, punct_opt, stop_word=False, lematizer=False, vectorizer=True, model=None, param_model=None):
    df = pd.read_csv(input_leg)  # Charger le fichier legislature dans un DataFrame
    dataset = complete_preproc(df, drop_fam=drop_fam, na_col=na_col, drop_names=drop_names, min_words=min_words, punct_opt=punct_opt)

    steps = []

    # Fonction de prétraitement personnalisée
    if stop_word:
        stop_words = set(stopwords.words('french'))
        steps.append(('stopwords', preprocess_text(text)))

    if lematizer:
        from nltk.stem import WordNetLemmatizer
        lemmatizer_ = WordNetLemmatizer()
        steps.append(('lemmatizer', Lemmatizer(lemmatizer_)))

    if vectorizer:
        tfidf = TfidfVectorizer(max_features=5000)  # Limite à 5000 mots les plus fréquents
        steps.append(('vectorizer', tfidf))

    X = dataset['Texte']  # Les textes à classifier
    y = dataset['famille']  # Les groupes politiques (labels)

    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ajouter le modèle au pipeline
    steps.append(('model', model(**param_model)))

    # Créer et ajuster le pipeline
    pipeline = Pipeline(steps)
    pipeline.fit(X_train, y_train)

    # Prédire sur l'ensemble de test
    y_pred = pipeline.predict(X_test)

    # Afficher le rapport de classification
    print(f"Rapport de classification pour le modèle {model.__name__} avec les paramètres suivants: {param_model}")
    print(classification_report(y_test, y_pred))
    class_report_ = classification_report(y_test, y_pred)
    return class_report_ '''
