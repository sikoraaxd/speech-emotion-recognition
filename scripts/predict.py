import torch
from model import SpeechEmotionClassifier
from dataset import AudioDataset
import sys
import os

if __name__ == '__main__':
    root_dir = './src'  if len(sys.argv) == 1 \
                        else sys.argv[1]
    
    dataset = AudioDataset(root_dir)
    model = SpeechEmotionClassifier()
    model.load_state_dict(torch.load('./scripts/model/speech_emotion_recognizer_model.pth'))

    preds = {}

    model.eval()
    for i, X in enumerate(dataset):
        filename = dataset.file_paths[i].as_posix()
        filename = filename.split('/')[-1]
        pred = model(X.unsqueeze(0)).argmax().item()
        preds[filename] = dataset.labels_meaning[pred]

    print(preds)
