import pandas as pd
import numpy as np
from assnat.params import *
from assnat.clean import complete_preproc
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import time, os
from collections.abc import Callable

leg_choice = {
    'leg15': ['data/leg15.csv'],
    'leg16': ['data/leg16.csv'],
    'legall': ['data/leg15.csv', 'data/leg16.csv']
}

label_encoder = LabelEncoder()

def load_X_y(leg_choice_key, min_n_words=30):
    preproc_csv = f"data/{leg_choice_key}-preproc.csv"
    if os.path.exists(preproc_csv):
        print(f'Loading data "{leg_choice_key}" from {preproc_csv}')
        df_preproc = pd.read_csv(preproc_csv)
    else:
        print(f'Loading data "{leg_choice_key}"')
        df = pd.DataFrame()
        for file in leg_choice[leg_choice_key]:
            df = pd.concat([df, pd.read_csv(file)], ignore_index=True, axis=0)
        df.reset_index(drop=True, inplace=True)

        df_preproc = complete_preproc(df, na_col=["Texte", "famille"], drop_names=["Mme la présidente", "M. le président"], min_words=min_n_words, punct_opt=True)
        print(f'Caching for next run to {preproc_csv}')
        df_preproc.to_csv(preproc_csv)

    X = df_preproc['Texte']  # Les textes à classifier
    y = df_preproc['famille']

    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print('Data loaded')

    return X_train, X_test, y_train, y_test



def tokenize_X(X_train, X_test, max_words = 100000):
    print('Tokenizing data')

    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, padding='post', truncating='post')

    print('Data tokenized')

    return X_train_pad, X_test_pad, len(tokenizer.word_index)

"""
[create_model] is a function that must look like this:

    def embedding_lstm( X_train, X_test, y_train, y_test):
        # Your code here
        return model,  X_train, X_test, y_train, y_test

[leg_choice_key] can be 'leg15', 'leg16' and 'all'
[min_n_words] to cut off speeches with less than n words
[patience, epochs, batch_size] are the usual parameters for the model fit
"""
def fit_predict(create_model : Callable[[any, any, any, any],(Model, any, any, any, any)],
                leg_choice_key,
                patience, epochs, batch_size,
                min_n_words=30):

    X_train, X_test, y_train, y_test = load_X_y(leg_choice_key, min_n_words=min_n_words)

    print('Creating model')
    model,  X_train, X_test, y_train, y_test = create_model(X_train, X_test, y_train, y_test)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    print('Training model')
    start_time = time.time()
    early_stopper = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopper],
        verbose=1
    )
    print(f'Model trained in {time.time() - start_time:,} seconds')

    print('Evaluating model')
    y_pred = model.predict(X_test)
    print('Model evaluated')

    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
