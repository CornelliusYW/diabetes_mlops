from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
dict_res = {0: 'Not-Diabetes', 1: 'Diabetes'}

pipeline_path = 'models/pipeline.pkl'
with open(pipeline_path, 'rb') as pipeline_file:
    pipeline = pickle.load(pipeline_file)

class DataInput(BaseModel):
    data: list

@app.post("/predict")
async def predict(input_data: DataInput):
    try:
        df = pd.DataFrame(input_data.data, columns=columns)
        predictions = pipeline.predict(df)
        results = [dict_res[pred] for pred in predictions]
    
        return {"predictions": results}
    
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
