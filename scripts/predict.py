import torch
from scripts.model import SpeechEmotionClassifier
from scripts.dataset import AudioDataset
import sys

def predict(root_dir: str = './src'):
    dataset = AudioDataset(root_dir)
    model = SpeechEmotionClassifier()
    model.load_state_dict(torch.load('./scripts/model/speech_emotion_recognizer_model.pth'))
    preds = []
    model.eval()
    for i, X in enumerate(dataset):
        filename = dataset.file_paths[i].as_posix()
        filename = filename.split('/')[-1]
        pred = model(X.unsqueeze(0)).argmax().item()
        pred = dataset.labels_meaning[pred]
        preds.append((filename, pred))

    if len(preds) == 1:
        return preds[0]
    else:
        return preds

if __name__ == '__main__':
    root_dir = './src'  if len(sys.argv) == 1 \
                        else sys.argv[1]
    
    preds = predict(root_dir=root_dir)

    print(preds)

