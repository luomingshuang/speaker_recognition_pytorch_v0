import random
import options as opt
import numpy as np

from audio_fbank import read_mfcc, sample_from_mfcc
from audio_stft import mfccs_and_spec

import torch
from model_resnet import DeepSpeakerModel

model = DeepSpeakerModel(embedding_size=opt.embedding_size,
                      num_classes=opt.classes)
#print(model)
# Reproducible results.
np.random.seed(123)
random.seed(123)

####using fbank
# Sample some inputs for WAV/FLAC files for the same speaker.
mfcc_001 = sample_from_mfcc(read_mfcc('samples/PhilippeRemy/lms_20200811174251.wav', opt.SAMPLE_RATE), opt.NUM_FRAMES)
print(mfcc_001.shape)

# Call the model to get the embeddings of shape (1, 512) for each file.
mfcc_001 = np.expand_dims(mfcc_001, axis=0)
print(mfcc_001.shape)
mfcc_001 = mfcc_001.transpose(0,3,1,2)

# numpy to tensor
mfcc_001 = torch.from_numpy(mfcc_001)
mfcc_001 = mfcc_001.clone().detach().float()
output_feature = model(mfcc_001)
output_classify = model.forward_classifier(mfcc_001)
print(output_feature.shape, output_classify.shape)

####using stft
stft_feature = mfccs_and_spec('samples/PhilippeRemy/lms_20200811174251.wav')
print('stft_feature shape: ', stft_feature.shape)
stft_feature = stft_feature.transpose(1,0)
stft_feature = np.expand_dims(stft_feature, axis=0)
stft_feature = np.expand_dims(stft_feature, axis=1)
print(stft_feature.shape)

stft = torch.from_numpy(stft_feature)
stft = stft.clone().detach().float()
output_feature = model(stft)
output_classify = model.forward_classifier(stft)
print(output_feature.shape, output_classify.shape)
