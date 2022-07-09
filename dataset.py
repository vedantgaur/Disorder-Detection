import os
import torch

from torch.utils.data import Dataset
import pandas as pd
import torchaudio

ANNOTATIONS_FILE = "C:\\Users\\vedan\\Downloads\\ml-stuttering-events-dataset\\SEP-28k_labels.csv"
AUDIO_DIR = "C:\\Users\\vedan\\Downloads\\ml-stuttering-events-dataset\\clip\\clips"
SAMPLE_RATE = 16000
NUM_SAMPLES = SAMPLE_RATE*3

class StutteringDataset(Dataset):
    def __init__(
        self, 
        annotations_file, 
        audio_dir, 
        transformation, 
        target_sample_rate, 
        num_samples, 
        device
    ) -> None:
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    # len(usd)
    def __len__(self):
        return len(self.annotations)

    # a_list[1] -> a_list.__getitem__(1)
    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr) # signal -> (num_of_channels, samples) -> (2, 16000) => (1, 16000)
        signal = self._mix_down_if_necessary(signal) # Changes to mono if necessary
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

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
    
    def _get_audio_sample_path(self, index):
        audio_title = f"{self.annotations.iloc[index, 0]}"
        # print(f"Audio Title: {audio_title}")
        folder_num = f"{self.annotations.iloc[index, 1]}"
        # print(f"Folder Num: {folder_num}")
        clip_name = f"{audio_title}_{folder_num}_{self.annotations.iloc[index, 2]}.wav"
        # print(f"Clip Name: {clip_name}")
        path = os.path.join(self.audio_dir, audio_title, folder_num, clip_name)
        # print(f"Path: {path}")
        return path
        
        
    def _get_audio_sample_label(self, index):
        # print(f"Label: {self.annotations.iloc[index, 13]}")
        if self.annotations.iloc[index, 13] != 3 and self.annotations.iloc[index, 13] != 2:
            return 1
        return 0



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

    sd = StutteringDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE, NUM_SAMPLES, device)
    print(f"There are {len(sd)} samples in the dataset.")
    for i in range(10):
        signal, label = sd[i]
        print(label)
    # sd._get_audio_sample_path(1)
    # sd._get_audio_sample_label(1)
    # print("--------------")
    # sd._get_audio_sample_path(3)
    # sd._get_audio_sample_label(3)
    # print(sd._get_audio_sample_label(1))
