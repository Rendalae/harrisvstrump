import pandas as pd
# $WIPE_BEGIN

from assnat.models import simple_logistic_regression
from assnat.clean import complete_preproc
import pickle
# $WIPE_END

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
#app.state.model = pickle.load(open('modelml.pkl', 'rb'))

# $WIPE_END
@app.get('/')
def root():
    model = pickle.load(open('modelml.pkl','rb'))
    assert model is not None
    params = model.get_params()
    print(f'here are the pRms {params}')
    return {'greetings':'hello'}

@app.get('/predict')
def predict(
    Texte: str,
    Theme: str
):
    model = pickle.load(open('modelml.pkl','rb'))
    assert model is not None

    X_test = pd.DataFrame(locals(), index=[0])
    print(X_test)
    print(X_test.columns)
    X_test.drop(columns=['model'],inplace=True)
    print(X_test)
    print(X_test.columns)

    X_test.columns = ['Texte', 'Thème Séance']

    print('X existing')
    X_processed = complete_preproc(X_test,simplify_fam=False)

    y_pred = model.predict(X_processed)
    print(y_pred)
    return dict(prediction = str(y_pred))

'''
@app.get('/predictproba')
def predict():
    model = pickle.load(open('modelml.pkl','rb'))
    assert model is not None
    X_test = pd.DataFrame({
    'Texte': ['Les riches ne sont pas assez taxés'],
    'Thème Séance': ['Taxes sur le revenu']
})
    X_processed = complete_preproc(X_test)
    y_pred = model.predict_proba(X_processed)
    print(y_pred)
    return dict(prediction = str(y_pred))
'''
@app.get('/predictproba')
def predict(
    Texte: str,
    Theme: str
):
    model = pickle.load(open('modelml.pkl','rb'))
    assert model is not None

    X_test = pd.DataFrame(locals(), index=[0])
    print(X_test)
    print(X_test.columns)
    X_test.drop(columns=['model'],inplace=True)
    print(X_test)
    print(X_test.columns)

    X_test.columns = ['Texte', 'Thème Séance']

    print('X existing')
    X_processed = complete_preproc(X_test, simplify_fam = False)
    y_pred = model.predict_proba(X_processed)
    classes = model.classes_
    result = {class_label: prob for class_label, prob in zip(classes, y_pred[0])}
    return result


#X_pred = pd.DataFrame(locals(), index = [0])

    #model = app.state.model
    #assert model is not None

#X_processed = complete_preproc(X_pred)
#y_pred = model.predict(X_processed)

    #return 'model loaded'
