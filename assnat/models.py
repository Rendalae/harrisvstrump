import pandas as pd
from assnat.clean import complete_preproc
from assnat.params import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pk


def simple_logistic_regression(legislation_number, merge_data_sets=True, drop_col = DROP_COLS, drop_fam = DROP_FAM, na_col = NA_COLS , drop_names = DROP_NAMES , min_words= MIN_WORDS, punct_opt=PUNCT_OPT):
    if merge_data_sets:
        data1 = pd.read_csv('data/leg15.csv')
        data2 = pd.read_csv('data/leg16.csv')
        df = pd.concat([data1, data2], ignore_index=True, axis=0)
    else:
        df = pd.read_csv(f'data/leg{legislation_number}.csv')

    df = complete_preproc(df, drop_col= drop_col, drop_fam= drop_fam, na_col= na_col, drop_names= drop_names, min_words = min_words, punct_opt= punct_opt)

    X_train, X_test, y_train, y_test = train_test_split(
        df[['Texte', 'Thème Séance']],
        df['famille'],
        test_size=0.2,
        random_state=42
    )

    tfidf_vectorizer = TfidfVectorizer(max_features=100000)

    preprocessor = ColumnTransformer(
        transformers=[
            ('tfidf_texte', tfidf_vectorizer, 'Texte'),
            ('tfidf_theme', tfidf_vectorizer, 'Thème Séance')
        ],
        remainder='passthrough'
    )

    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(multi_class='multinomial', max_iter=100000))
    ])

    model_pipeline.fit(X_train, y_train)

    y_pred = model_pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print(classification_report(y_test, y_pred))
    pk.dump(model_pipeline, open('/Users/aaronviviani/code/aaronviviani/08\ -\ Project ', 'wb'))
    return accuracy
