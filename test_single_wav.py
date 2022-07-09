import shutil

import torch
import torch.nn as nn
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
import argparse

from model import CNNNetwork
# from process_data import SAMPLE_RATE, NUM_SAMPLES, AudioSample
from process_data1 import get_signal

from fastapi import FastAPI, UploadFile, File
import uvicorn

app = FastAPI()

def save_file(filename, data):
    with open(filename, 'wb') as f:
        f.write(data)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    contents = await file.read()
    save_file(file.filename, contents)
    return {"Filename": file.filename}

# uvicorn test_single_wav:app --reload

class_mapping = [
    "no_disorder",
    "disorder",
]

model_path = "nets/sd_cnn1.pth"
wav_path = "wavs/HeStutters_0_27.wav"


def predict(model, input, class_mapping):
    model.eval() # If eval() activated, layers like dropout and batch normalization are turned off
                 # train() reverts that (on and off switch)
    with torch.no_grad():
        prediction = model(input) # Tensor(num of samples, num of classes -> 10) -> confidence
        predicted_index = prediction[0].argmax(0)
        predicted = class_mapping[predicted_index] # Predicted class
    return predicted, prediction

def run():
    cnn = CNNNetwork()
    # state_dict = torch.load("nets/sd_cnn1.pth")
    state_dict = torch.load(model_path)
    cnn.load_state_dict(state_dict)
    
    input = get_signal(wav_path)
    input.unsqueeze_(0)
    

    # Make an inference
    predicted, prediction = predict(cnn, input, class_mapping) #Class prediction, Confidence
    return predicted, prediction

if __name__ == "__main__":
    p0, p1 = run()
    print(f"Class: {p0}")
    print(f"Confidence: {p1}")