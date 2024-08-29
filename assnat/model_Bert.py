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


def train_Bert(leg_, min_words_, na_col_, simplify_fam_, drop_names_, drop_fam_, punct_opt_, sample_size_, batch_size_, patience_, epoch_, max_len_, save_id_):

    # Load and preprocess data
    if leg_ == 'all':
        data1 = pd.read_csv('data/leg15.csv')
        data2 = pd.read_csv('data/leg16.csv')
        df = pd.concat([data1, data2], ignore_index=True, axis=0)
    else:
        df = pd.read_csv(leg_)
    df_preproc = complete_preproc(df, na_col=na_col_, simplify_fam= simplify_fam_, drop_fam=drop_fam_, drop_names=drop_names_, min_words=min_words_, punct_opt=punct_opt_)
    print("Data loaded and preprocessed!")

    df_preproc = df_preproc.sample(n=sample_size_, random_state=42).reset_index(drop=True)

    X = df_preproc['Texte']  # Les textes à classifier
    y = df_preproc['famille']

    # Load Bert tokenizer
    tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base", padding_side = "right")
    model = TFAutoModel.from_pretrained("almanach/camembert-base", from_pt = True)
    print("Camembert tokenizer and model loaded!")

    # Embedd by processing the data in smaller batches
    tokenized_tensors = tokenizer(X.tolist(), max_length=max_len_, padding = "max_length", truncation = True, return_tensors="tf")
    embeddings = []
    nb_batches = int(round(sample_size_ / batch_size_,0))
    for i in range(0, sample_size_, batch_size_):
        batch_tensors = {k: v[i:i+batch_size_] for k, v in tokenized_tensors.items()}
        batch_embeddings = model.predict(batch_tensors)
        embeddings.append(batch_embeddings)
    X_embed = np.concatenate([embeddings[x].last_hidden_state[:,0,:] for x in range(nb_batches)], axis=0)

    # Create labels
    label_encoder = LabelEncoder()
    y_target = label_encoder.fit_transform(y)
    label_dim = len(label_encoder.classes_)

    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_embed, y_target, test_size=0.2, random_state=42)
    print(f"Labels and features ready, with {label_dim} classes!")

    # Create model
    input_shape = (768,)
    dense_model = Sequential([
    Dense(256, activation='tanh', input_shape=input_shape),
    Dense(128, activation='relu'),
    Flatten(),
    Dense(label_dim, activation='softmax')
    ])

    # Train and fit model
    early_stopper = EarlyStopping(monitor='val_loss', patience=patience_, restore_best_weights=True)
    dense_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    # checkpoints_dir=f'data/checkpoints/{timestamp()}'
    # os.makedirs(checkpoints_dir, exist_ok=True)
    # checkpoints_path = checkpoints_dir+"/cp-{epoch:04d}.ckpt"
    # checkpoints = ModelCheckpoint(
    #     filepath=checkpoints_path,
    #     save_weights_only=True,
    #     save_freq='epoch')
    dense_model.fit(X_train, y_train, validation_split=0.2, epochs=epoch_, batch_size=32, callbacks=[early_stopper])
    print("Model trained!")

    # Save the entire model to a HDF5 file.
    models_dir=f'models/{timestamp()}'
    os.makedirs(models_dir, exist_ok=True)
    model_name = f"model_bert_{save_id_}.h5"
    # The '.h5' extension indicates that the model should be saved to HDF5.
    dense_model.save(model_name)
    print("Model saved!")
    #save_model(dense_model, model_path, overwrite=True)

    #dense_model.evaluate(X_test, y_test)

    # Prédire sur l'ensemble de test
    y_pred = dense_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Afficher le rapport de classification
    print("Rapport de classification pour le Dense avec Bert:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

    pass

# Parameters
leg_1 = 'all'
min_words_1=10
na_col_1=["Texte", "famille"]
drop_names_1=["Mme la présidente", "M. le président"]
drop_fam_1 = ["Variable"]
simplify_fam_1=True
punct_opt_1=True
sample_size_1 = 2_000
max_len_1 = 50
batch_size_1=1000
patience_1=5
epoch_1=10
save_id_1 = "all_4"

# Execute function
train_Bert(leg_=leg_1
           , min_words_=min_words_1
           , na_col_=na_col_1
           , drop_names_=drop_names_1
           , drop_fam_=drop_fam_1
           , simplify_fam_=simplify_fam_1
           , punct_opt_=punct_opt_1
           , sample_size_=sample_size_1
           , batch_size_=batch_size_1
           , patience_=patience_1
           , epoch_=epoch_1
           , max_len_=max_len_1
           ,save_id_=save_id_1)
