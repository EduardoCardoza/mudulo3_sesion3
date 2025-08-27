import uvicorn
from fastapi import FastAPI, File, UploadFile
from io import StringIO
import pandas as pd
from joblib import load

app = FastAPI()



@app.get("/")
def read_root():
    return {"message": "Hello"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict_bancknote(file: UploadFile = File(...)):
    classifier = load("linear_regression.joblib")
    #cargamos el modelo entrenado, usaremos para predicicon del modelo
    #recibimos el csv a dataframe y pasamos el contenido de la columna de nombre 0 a una lista llamada features
    features_df = pd.read_csv('selected_features.csv')
    features = features_df['0'].to_list()

    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))
    #leemos el archivo que sube y lo decodificamos y pasamos a dataframe
    
    df = df[features] #delimitamos las columnas del df cargado con las de caracteristicas para que sean las mismas y se pueda inferir

    predictions = classifier.predict(df)
    
    return {
        "predictions": predictions.tolist()
    }
