from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn

# Define request model
class InputData(BaseModel):
    inputs: list

# Load the model class definition
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the FastAPI app
app = FastAPI()

# Load the model
model = SimpleNN()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Define the prediction endpoint
@app.post("/predict")
async def predict(input_data: InputData):
    try:
        # Convert input data to torch tensor
        inputs = torch.tensor(input_data.inputs, dtype=torch.float32)
        
        # Check the input dimensions
        if inputs.dim() != 1 or inputs.size(0) != 10:
            raise HTTPException(status_code=400, detail="Input must be a 1D array with 10 elements.")
        
        # Make prediction
        with torch.no_grad():
            prediction = model(inputs).item()
        
        return {"prediction": prediction}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server (for local testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
