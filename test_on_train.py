import torch
import torch.nn as nn
import torchaudio
import numpy as np
import argparse

from model import CNNNetwork
from dataset import StutteringDataset, AUDIO_DIR, ANNOTATIONS_FILE, SAMPLE_RATE, NUM_SAMPLES

class_mapping = [
    "no_disorder",
    "disorder",
]


def predict(model, input, target, class_mapping):
    model.eval() # If eval() activated, layers like dropout and batch normalization are turned off
                 # train() reverts that (on and off switch)
    with torch.no_grad():
        prediction = model(input) # Tensor(num of samples, num of classes -> 10)
        predicted_index = prediction[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    cnn = CNNNetwork()
    # state_dict = torch.load("nets/sd_cnn1.pth")
    state_dict = torch.load("nets/" + args.path)
    cnn.load_state_dict(state_dict)

    # Load the MNIST validation dataset
    # Instantiating dataset object
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=1024, 
        hop_length=512, 
        n_mels=64
    )

    sd = StutteringDataset(
        ANNOTATIONS_FILE, 
        AUDIO_DIR, 
        mel_spectrogram, 
        SAMPLE_RATE, 
        NUM_SAMPLES, 
        device="cpu")

    sd_len = len(sd)
    sd_arr = np.zeros(sd_len)


    # Get a sample from the urban sound dataset for inference
    for i in range(len(sd)):
        input, target = sd[i][0], sd[i][1] # [batch size, num_channels, fr, time]
        input.unsqueeze_(0)
        
    

        # Make an inference
        predicted, expected = predict(cnn, input, target, class_mapping)
        print(f"Predicted: '{predicted}', expected: '{expected}'")
        if (predicted == expected):
            sd_arr[i] = 1
    
    print(f"\n\n\nMEAN:{np.mean(sd_arr)}")
