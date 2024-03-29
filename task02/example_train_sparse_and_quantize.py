from pathlib import Path
from tqdm.auto import tqdm
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from torchvision.models import resnet101
# from sparseml.pytorch.datasets import CIFAR10
from torchvision.datasets import CIFAR10
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import export_onnx

def save_onnx(model, export_path, convert_qat):
    # It is important to call torch_model.eval() or torch_model.train(False) before exporting the model, to turn the model to inference mode.
    # This is required since operators like dropout or batchnorm behave differently in inference and training mode.
    model.eval()
    sample_batch = torch.randn((1, 3, 224, 224))
    export_onnx(model, sample_batch, export_path, convert_qat=convert_qat)


def parse_args():
    parser = argparse.ArgumentParser(description='Specify task parameters')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training (default: 256)')
    parser.add_argument('--checkpoint', type=str, help='Path to the saved teacher\'s weights')
    parser.add_argument('--logging_step', type=int, help='Logging step', default=20)
    parser.add_argument('--use_mse_during_distillation', type=str, help='Defines whether to use MSE during distillation', default=False)
    args = parser.parse_args()
    return args

def train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device):
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
        
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss.item() / len(train_loader.dataset)
    epoch_acc = running_corrects.double().item() / len(train_loader.dataset)
    return epoch_loss, epoch_acc


def evaluate(model,
             test_loader,
             device):
    model.eval()
    running_corrects = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    return running_corrects.double().item() / len(test_loader.dataset)
    

def main():
    # TODO: add argparse/hydra/... to manage hyperparameters like batch_size, path to pretrained model, etc
    args = parse_args()

    # Sparsification recipe -- yaml file with instructions on how to sparsify the model
    recipe_path = "recipe.yaml"
    assert Path(recipe_path).exists(), "Didn't find sparsification recipe!"

    checkpoints_path = Path("checkpoints")
    checkpoints_path.mkdir(exist_ok=True)

    # Model creation
    NUM_CLASSES = 10  # number of CIFAR10 classes
    model = resnet101(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.layer3 = torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    model.load_state_dict(torch.load(args.checkpoint))
    

    save_onnx(model, checkpoints_path / "baseline_resnet.onnx", convert_qat=False)

    # Dataset creation
    transforms_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
    test_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Loss setup
    criterion = nn.CrossEntropyLoss()
    # Note that learning rate is being modified in `recipe.yaml`
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # SparseML Integration
    manager = ScheduledModifierManager.from_yaml(recipe_path)
    optimizer = manager.modify(model, optimizer, steps_per_epoch=len(train_loader))

    # Training Loop
    pbar = tqdm(range(manager.max_epochs), desc="epoch")
    for epoch in pbar:
        epoch_loss, epoch_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        epoch_test_acc = evaluate(model, test_loader, device)
        pbar.set_description(f"Training loss: {epoch_loss:.3f}  Training accuracy: {epoch_acc:.3f}  Testing accuracy: {epoch_test_acc:.3f}")

    manager.finalize(model)

    # Saving model
    save_onnx(model, checkpoints_path / "pruned_quantized_resnet.onnx", convert_qat=True)

if __name__ == "__main__":
    main()