import torch
from mypath import Path
import numpy as np
from network import C3D_model
import cv2
torch.backends.cudnn.benchmark = True

def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Devices available:", torch.cuda.device_count())
    print("Using device:", device)

    with open(Path.ucf_labels(), 'r') as f:
        class_names = f.readlines()
        f.close()

    model = C3D_model.C3D(num_classes=101, pretrained=True)
    model.to(device)
    model.eval()

    cap = cv2.VideoCapture(Path.inference_video())
    retaining = True

    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
        if len(clip) == 16:
            inputs = np.array(clip).astype(np.float32)
            inputs = np.expand_dims(inputs, axis=0)
            inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
            inputs = torch.from_numpy(inputs)
            inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            # Print the label and probability
            print("Label: ", class_names[label].split(' ')[-1].strip())
            print("Probability: %.4f" % probs[0][label])

            clip.pop(0)

    cap.release()

if __name__ == '__main__':
    main()
