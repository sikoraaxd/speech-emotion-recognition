import torch
import torchaudio
import numpy as np

from typing import Tuple



def open(audio_file: str) -> Tuple[torch.Tensor, int]:
    sig, sr = torchaudio.load(audio_file)
    return sig, sr


def rechannel(aud: Tuple[torch.Tensor, int], new_channel: int) -> Tuple[torch.Tensor, int]:
    sig, sr = aud

    if (sig.shape[0] == new_channel):
        return aud

    if (new_channel == 1):
        resig = sig[:1, :]
    else:
        resig = torch.cat([sig, sig])

    return resig, sr


def resample(aud: Tuple[torch.Tensor, int], newsr: int) -> Tuple[torch.Tensor, int]:
    sig, sr = aud

    if (sr == newsr):
        return aud

    num_channels = sig.shape[0]
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
        retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
        resig = torch.cat([resig, retwo])

    return resig, newsr


def pad_trunc(aud: Tuple[torch.Tensor, int], max_ms: int) -> Tuple[torch.Tensor, int]:
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
        sig = sig[:,:max_len]

    elif (sig_len < max_len):
        pad_begin_len = np.random.randint(0, max_len - sig_len)
        pad_end_len = max_len - sig_len - pad_begin_len

        pad_begin = torch.zeros((num_rows, pad_begin_len))
        pad_end = torch.zeros((num_rows, pad_end_len))

        sig = torch.cat((pad_begin, sig, pad_end), 1)
    
    return sig, sr


def time_shift(aud: Tuple[torch.Tensor, int], shift_limit: float) -> Tuple[torch.Tensor, int]:
    sig, sr = aud
    _, sig_len = sig.shape
    shift_amt = int(np.random.random() * shift_limit * sig_len)
    return sig.roll(shift_amt), sr


def spectrogram(aud: Tuple[torch.Tensor, int], n_mels: int=64, n_fft: int=1024, hop_len: int=None) -> torch.Tensor:
    sig,sr = aud
    top_db = 80

    spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
    spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
    return spec


def spectro_augment(spec: torch.Tensor, max_mask_pct: float=0.1, n_freq_masks: int=1, n_time_masks: int=1) -> torch.Tensor:
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
        aug_spec = torchaudio.transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
        aug_spec = torchaudio.transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec