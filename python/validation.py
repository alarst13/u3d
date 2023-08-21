import timeit
from tqdm import tqdm
import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
from video_classification.dataloaders.dataset import CombinedDataset
import argparse
import torch
from video_classification.network import C3D_model
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

# Use GPU if available else revert to CPU
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)


def test_model(model, criterion, test_dataloader):
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    test_size = len(test_dataloader.dataset)

    all_labels = []
    all_preds = []

    start_time = timeit.default_timer()
    for inputs, labels in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs, _ = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        preds = torch.max(probs, 1)[1]
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    epoch_loss = running_loss / test_size
    epoch_acc = running_corrects.double() / test_size

    print("Loss: {} Acc: {}".format(epoch_loss, epoch_acc))
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")

    return all_labels, all_preds


def calculate_attack_success_rate(original_preds, perturbed_preds):
    """
    Calculate the success rate of adversarial attacks based on predictions.

    Given the original predictions and the predictions after perturbation, this function
    calculates the rate at which the perturbations were successful in causing the model
    to make a different prediction.

    The success rate is computed as:

    Success Rate = (Number of instances where original_preds[i] ≠ perturbed_preds[i]) / Total number of instances

    Args:
    - original_preds (list or array): The list of predictions made by the model on the original, unperturbed data.
    - perturbed_preds (list or array): The list of predictions made by the model on the perturbed (or attacked) data.

    Returns:
    - float: The success rate of the attacks.
    """
    successful_attacks = sum(1 for o, p in zip(
        original_preds, perturbed_preds) if o != p)
    return successful_attacks / len(original_preds)


def main(args):
    dataset = args.dataset
    model_path = args.model

    dataset = 'ucf101' if args.dataset == 'u' else 'hmdb51'
    if dataset == 'ucf101':
        num_classes = 101
    elif dataset == 'hmdb51':
        num_classes = 51
    else:
        raise ValueError("Unsupported dataset name")

    # init model
    model = C3D_model.C3D(num_classes=num_classes)
    checkpoint = torch.load(
        model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    org_dataloader = DataLoader(CombinedDataset(
        data_dir=args.data_org, dataset=dataset, clip_len=16), batch_size=20, num_workers=4)
    prt_dataloader = DataLoader(CombinedDataset(
        data_dir=args.data_org, dataset=dataset, clip_len=16), batch_size=20, num_workers=4)

    print("Testing on original data")
    original_preds = test_model(model, criterion, org_dataloader)
    print("Testing on perturbed data")
    perturbed_preds = test_model(model, criterion, prt_dataloader)

    success_rate = calculate_attack_success_rate(
        original_preds, perturbed_preds)
    print(f"Attack success rate: {success_rate * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Video classification using C3D model')

    parser.add_argument('--dataset', type=str, choices=[
                        'u', 'h'], default='u', help='Dataset name: "u" for UCF101 or "h" for HMDB51')

    parser.add_argument('--data_org', type=str, required=True,
                        help='Path to the original data splits')

    parser.add_argument('--data_prt', type=str, required=True,
                        help='Path to the perturbed data splits')

    parser.add_argument('--model', type=str, required=True,
                        help='Path to the model checkpoint')

    args = parser.parse_args()

    main(args)
