from fastapi import FastAPI, Request
import pandas as pd
import json, io
from fastapi.middleware.cors import CORSMiddleware
from io import StringIO
from inference.infernce_model import run_inference

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify domains like ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock model for demonstration
class Model:
    def predict(self, data, model_code):
        pred = run_inference(data)
        # remove last element
        # pred = pred[:-1]
        
        # Mapping with data 
        for i in range(len(pred)):
            pred[i]["keid"] = str(data.iloc[i, 0])
            pred[i]["kepler_name"] = str(data.iloc[i, 1])
        return pred

model = Model()

def csv_to_dict(df, n) -> dict:
    result = {}
    try:
        for row in df.iterrows():
            if len(row) > n and str(row[0]).strip():  # ignore empty keys
                result[str(row[0]).strip()] = str(row[n]).strip()
    except Exception as e:
        print(f"Error processing row: {e}")
    return result

@app.post("/predict")
async def predict(request: Request):
    body = await request.body()
    csv_text = body.decode("utf-8")
    df = pd.read_csv(StringIO(csv_text), delimiter=",", lineterminator="\n", skip_blank_lines=True)
    df = df.drop(index=0)
    df = df.drop(df.columns[1], axis=1)
    df = df.transpose()
    df.reset_index(drop=True, inplace=True)
    df = df.drop(index=0)
    df.columns = range(df.shape[1])  # remove column names
    df = df.dropna(how="all")
    # Make first row as header

    data_dict = [csv_to_dict(df, i) for i in range(df.shape[1])]
    import json
    with open('data.json', 'w') as f:
        json.dump(data_dict, f, indent=4)

    predictions = model.predict(df, "model-code")
    return predictions
