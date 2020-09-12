import librosa
import numpy as np
import torch
import torch.autograd as grad
import torch.nn.functional as F

import options as opt

def mfccs_and_spec(wav_file, wav_process = True, calc_mfccs=False, calc_mag_db=False):    
    sound_file, _ = librosa.core.load(wav_file, sr=opt.sr)
    if len(sound_file) ==0:
        print(wav_file, len(sound_file), int(opt.window*opt.sr))
    window_length = int(opt.window * opt.sr)
    hop_length = int(opt.hop * opt.sr)
    duration = opt.tisv_frame * opt.hop + opt.window
    
    # Cut silence and fix length
    if wav_process == True:
        sound_file, index = librosa.effects.trim(sound_file, frame_length=window_length, hop_length=hop_length)
        length = int(opt.sr * duration)
        sound_file = librosa.util.fix_length(sound_file, length)
        
    spec = librosa.stft(sound_file, n_fft=opt.nfft, hop_length=hop_length, win_length=window_length)
    mag_spec = np.abs(spec)

    return mag_spec