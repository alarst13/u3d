import argparse
import torch
import numpy as np
from network import C3D_model
import cv2
torch.backends.cudnn.benchmark = True


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def main(video_path, dataset, model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)

    dataset = 'ucf101' if args.dataset == 'u' else 'hmdb51'
    if dataset == 'ucf101':
        labels_path = 'dataloaders/ucf_labels.txt'
        num_classes = 101
    elif dataset == 'hmdb51':
        labels_path = 'dataloaders/hmdb_labels.txt'
        num_classes = 51
    else:
        raise ValueError("Unsupported dataset name")

    with open(labels_path, 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model
    model = C3D_model.C3D(num_classes=num_classes)
    checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # read video
    video = video_path
    cap = cv2.VideoCapture(video)
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
            inputs = torch.autograd.Variable(
                inputs, requires_grad=False).to(device)
            with torch.no_grad():
                outputs, _ = model.forward(inputs)

            probs = torch.nn.Softmax(dim=1)(outputs)
            label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]

            print(class_names[label].split(' ')[-1].strip())

            clip.pop(0)

    cap.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Video classification using C3D model')
    parser.add_argument('--video', type=str,
                        required=True, help='Path to the input video')
    parser.add_argument('--dataset', type=str, choices=[
                        'u', 'h'], default='u', help='Dataset name: "u" for UCF101 or "h" for HMDB51')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the model checkpoint')

    args = parser.parse_args()

    main(args.video, args.dataset, args.model)
