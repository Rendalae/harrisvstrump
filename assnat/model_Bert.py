import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Flatten, SpatialDropout1D, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential, load_model, save_model

from transformers import AutoTokenizer
from transformers import TFAutoModel

from assnat.params import *
from assnat.clean import complete_preproc
from assnat.utils import timestamp

import time, os
from collections.abc import Callable


def train_Bert(leg_, min_words_, na_col_, punct_opt_, sample_size_, batch_size_, patience_, epoch_, max_len_):

    # Load and preprocess data
    df = pd.read_csv(leg_)
    df_preproc = complete_preproc(df, na_col=["Texte", "famille"], simplify_fam= True, drop_fam=['Variable'], drop_names=["Mme la présidente", "M. le président"], min_words=min_words_, punct_opt=True)
    print("Data loaded and preprocessed!")

    X = df_preproc['Texte']  # Les textes à classifier
    y = df_preproc['famille']

    # Load Bert tokenizer
    tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base", padding_side = "right")
    model = TFAutoModel.from_pretrained("almanach/camembert-base", from_pt = True)
    print("Camembert tokenizer and model loaded!")

    # Embedd by processing the data in smaller batches
    tokenized_tensors = tokenizer(X[0:sample_size_].tolist(), max_length=max_len_, padding = "max_length", truncation = True, return_tensors="tf")
    embeddings = []
    nb_batches = int(round(sample_size_ / batch_size_,0))
    for i in range(0, sample_size_, batch_size_):
        batch_tensors = {k: v[i:i+batch_size_] for k, v in tokenized_tensors.items()}
        batch_embeddings = model.predict(batch_tensors)
        embeddings.append(batch_embeddings)
    embeddings_all = np.concatenate([embeddings[x].last_hidden_state[:,0,:] for x in range(nb_batches)], axis=0)

    # Create labels and features
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    label_dim = len(label_encoder.classes_)
    X_sample = embeddings_all
    y_sample = y_encoded[0:sample_size_]
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
    print(f"Labels and features ready, with {label_dim} classes!")

    # Create model
    input_shape = (768,)
    dense_model = Sequential([
    Dense(256, activation='relu', input_shape=input_shape),
    Flatten(),
    Dense(label_dim, activation='softmax')
    ])

    # Train and fit model
    early_stopper = EarlyStopping(monitor='val_loss', patience=patience_, restore_best_weights=True)
    dense_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    checkpoints_dir=f'data/checkpoints/{timestamp()}'
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoints_path = checkpoints_dir+"/cp-{epoch:04d}.ckpt"
    checkpoints = ModelCheckpoint(
        filepath=checkpoints_path,
        save_weights_only=True,
        save_freq='epoch')
    dense_model.fit(X_train, y_train, validation_split=0.2, epochs=epoch_, batch_size=32, callbacks=[early_stopper,checkpoints])
    print("Model trained!")

    # Save model
    models_dir=f'data/models/{timestamp()}'
    os.makedirs(models_dir, exist_ok=True)
    model_path = models_dir+"/bert-{epoch:04d}.keras"
    save_model(dense_model, model_path, overwrite=True)
    print("Model saved!")

    #dense_model.evaluate(X_test, y_test)

    # Prédire sur l'ensemble de test
    y_pred = dense_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Afficher le rapport de classification
    print("Rapport de classification pour le Dense avec Bert:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

    pass

# Parameters
leg_ = 'data/leg16.csv'
min_words_=10
na_col_=["Texte", "famille"]
drop_names_=["Mme la présidente", "M. le président"]
punct_opt_=True
sample_size_ = 2_000
max_len_ = 50
batch_size_=1000
patience_=5
epoch_=5

# Execute function
train_Bert(leg_, min_words_, na_col_, punct_opt_, sample_size_, batch_size_, patience_, epoch_, max_len_)
