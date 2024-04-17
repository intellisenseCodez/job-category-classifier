from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from joblib import load
import sklearn


# Define a FastAPI app
app = FastAPI()

# Create a model for the request data
class UserSearch(BaseModel):
    search: str  # Define the expected data type of the input value
    
# Define the POST endpoint
@app.post("/api/predict")
async def predict_category(user_search: UserSearch):
    # Process the input data and return a JSON response
    try:
        search = user_search.search
        
        # perform prdiction on the search
        model_filename = "./classifier/model/model.joblib"
        with open(model_filename, 'rb') as f_in:
            model = load(f_in)
    
        pred = model.predict([search])
        

        # Return the output in JSON format
        response_data = {
            "User Search": search,
            "Prediction": pred[0]
        }
        
        return JSONResponse(content=response_data)

    except Exception as e:
        # If there's any error, return a JSON error response
        raise HTTPException(status_code=400, detail=str(e))
