import torch
from torch.utils.data import Dataset
import pathlib
import os
import util


class AudioDataset(Dataset):
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = pathlib.Path(dataset_path)
        self.file_paths = []

        self.duration = 10000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

        self.labels_meaning = {
            0: 'нейтрально',
            1: 'спокойно',
            2: 'счастливо',
            3: 'грустно',
            4: 'сердито',
            5: 'напуганно',
            6: 'недовольно',
            7: 'удивлённо'
        }

        for audio in os.listdir(self.dataset_path):
            audiopath = self.dataset_path.joinpath(audio)
            self.file_paths.append(audiopath)


    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        aud = util.open(self.file_paths[idx])
  
        reaud = util.resample(aud, self.sr)
        rechan = util.rechannel(reaud, self.channel)

        dur_aud = util.pad_trunc(rechan, self.duration)
        shift_aud = util.time_shift(dur_aud, self.shift_pct)
        sgram = util.spectrogram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = util.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram