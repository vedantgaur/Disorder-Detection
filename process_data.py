import os
import torch

from torch.utils.data import Dataset
import pandas as pd
import torchaudio

SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE*7 # 7 seconds

class AudioSample(Dataset):
    def __init__(
        self,  
        transformation, 
        target_sample_rate, 
        num_samples, 
        device,
        audio_path
    ) -> None:
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.audio_path = audio_path



    # a_list[1] -> a_list.__getitem__(1)
    def __getitem__(self):
        audio_sample_path = self.audio_path
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr) # signal -> (num_of_channels, samples) -> (2, 16000) => (1, 16000)
        signal = self._mix_down_if_necessary(signal) # Changes to mono if necessary
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal

    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1: # (2, 1000) -> 2 > 1
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

def process(audio_path):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, 
            n_fft=1024, 
            hop_length=512, 
            n_mels=64
        )
    sd = AudioSample(mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device, audio_path)
    return sd

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=1024, 
        hop_length=512, 
        n_mels=64
    )
    # ms = mel_spectrogram(signal)

    std = AudioSample(mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device, "wavs/wav1.wav")
    signal = std
    print(std)