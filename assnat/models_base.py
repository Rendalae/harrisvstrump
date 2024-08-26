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
import time
from collections.abc import Callable

leg_choice = {
    'leg15': ['data/leg15.csv'],
    'leg16': ['data/leg16.csv'],
    'all': ['data/leg15.csv', 'data/leg16.csv']
}

label_encoder = LabelEncoder()

def load_X_y(leg_choice_key, min_n_words=30):
    print('Loading data for', leg_choice_key)

    df = pd.DataFrame()
    for file in leg_choice[leg_choice_key]:
        df = pd.concat([df, pd.read_csv(file)], ignore_index=True, axis=0)
    df.reset_index(drop=True, inplace=True)

    df_preproc = complete_preproc(df, na_col=["Texte", "famille"], drop_names=["Mme la présidente", "M. le président"], min_words=min_n_words, punct_opt=True)

    X = df_preproc['Texte']  # Les textes à classifier
    y = df_preproc['famille']

    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print('Data loaded')

    return X_train, X_test, y_train, y_test



def tokenize_X(X_train, X_test, y_train, y_test, max_words = 5000, pad_max_len=200):
    print('Tokenizing data')

    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=pad_max_len, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=pad_max_len, padding='post', truncating='post')

    print('Data tokenized')

    return X_train_pad, X_test_pad, y_train, y_test



def fit_predict(create_model : Callable[[any, any],Model], leg_choice_key, patience, epochs, batch_size):

    X_train, X_test, y_train, y_test = tokenize_X(*load_X_y(leg_choice_key))

    print('Creating model')
    model = create_model(X_train, y_train)
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
        verbose=2
    )
    print(f'Model trained in {time.time() - start_time:,} seconds')

    print('Evaluating model')
    y_pred = model.predict(X_test)
    print('Model evaluated')

    y_pred_classes = np.argmax(y_pred, axis=1)
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))
