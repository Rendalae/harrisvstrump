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

@app.get('/')
def predict():
    model = pickle.load(open('modelml.pkl','rb'))
    assert model is not None
    #X_test = pd.DataFrame({'Texte': 'Je ne suis pas)

#X_pred = pd.DataFrame(locals(), index = [0])

    #model = app.state.model
    #assert model is not None

#X_processed = complete_preproc(X_pred)
#y_pred = model.predict(X_processed)

    #return 'model loaded'
