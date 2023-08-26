import timeit
import os
import glob
from tqdm import tqdm
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import C3D_model

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

modelName = 'C3D'


def train_model(org_data_path, data_path, dataset, save_dir, num_classes, lr, num_epochs, save_epoch, useTest, test_interval, saveName, resume_epoch):
    """
    Args:
        num_classes (int): Number of classes in the data
        num_epochs (int, optional): Number of epochs to train for.
    """

    # C3D model
    model = C3D_model.C3D(num_classes=num_classes,
                          pretrained=True, pretrained_model=args.pretrained)
    train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                    {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]

    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
                                map_location=lambda storage, loc: storage)   # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel()
          for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    print('Training model on {} dataset...'.format(dataset))

    train_dataloader = DataLoader(VideoDataset(root_dir=org_data_path, output_dir=data_path,
                                  dataset=dataset, split='train', clip_len=16), batch_size=20, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(VideoDataset(root_dir=org_data_path, output_dir=data_path,
                                dataset=dataset, split='val',  clip_len=16), batch_size=20, num_workers=4)
    test_dataloader = DataLoader(VideoDataset(root_dir=org_data_path, output_dir=data_path,
                                 dataset=dataset, split='test', clip_len=16), batch_size=20, num_workers=4)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in [
        'train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs, _ = model(inputs)
                else:
                    with torch.no_grad():
                        outputs, _ = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase,
                      epoch+1, num_epochs, epoch_loss, epoch_acc))
            else:
                print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase,
                      epoch+1, num_epochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(
                save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

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

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch +
                  1, num_epochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")


def main(args):
    dataset_name = 'ucf101' if args.dataset == 'u' else 'hmdb51'
    if dataset_name == 'hmdb51':
        num_classes = 51
    elif dataset_name == 'ucf101':
        num_classes = 101

    save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))

    resume_epoch = args.resume_epoch

    if resume_epoch != 0:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) if runs else 0
    else:
        runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

    save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
    saveName = modelName + '-' + dataset_name

    nEpochs = args.nEpochs
    useTest = args.useTest
    nTestInterval = args.nTestInterval
    snapshot = args.snapshot
    lr = args.lr
    # The directory where the preprocessed dataset is saved
    data_path = args.data_splits
    # The directory where the original dataset is saved
    # This is used to check the integrity of the preprocessed dataset
    org_data_path = args.data_org

    train_model(org_data_path=org_data_path, data_path=data_path, dataset=dataset_name, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval, saveName=saveName, resume_epoch=resume_epoch)


if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Train C3D model')

    # Add command-line arguments with default values
    parser.add_argument('--dataset', type=str, choices=[
                        'u', 'h'], required=True, help='Dataset name: "u" for UCF101 or "h" for HMDB51')
    parser.add_argument('--nEpochs', type=int, default=20,
                        help='Number of epochs for training')
    parser.add_argument('--resume_epoch', type=int, default=0,
                        help='Epoch to resume training from')
    parser.add_argument('--useTest', type=bool, default=True,
                        help='Whether to perform test evaluation')
    parser.add_argument('--nTestInterval', type=int,
                        default=20, help='Test evaluation interval')
    parser.add_argument('--snapshot', type=int, default=10,
                        help='Checkpoint saving interval')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for optimization')
    parser.add_argument('--data_org', type=str, required=True,
                        help='Path to the directory where the original dataset is saved')
    parser.add_argument('--data_splits', type=str, required=True,
                        help='Path to the directory where the preprocessed dataset was saved')
    parser.add_argument('--pretrained', '-p', type=str, required=True,
                        help='Path to pretrained model for finetuning')

    # Parse the command-line arguments
    args = parser.parse_args()

    main(args)
