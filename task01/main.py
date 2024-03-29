import argparse
import numpy as np
import random
import torch
import torchvision
import wandb

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet101, ResNet101_Weights

from training import train, distill

SEED = 0xDEADBEEF


def parse_args():
    parser = argparse.ArgumentParser(description='Specify task parameters')
    parser.add_argument('--task', type=str, default="finetune",
                        help='Task type: \'finetune\' for finetuning, \'distill\' for distillation')
    parser.add_argument('--model_type', type=str, default='big',
                        choices=['big', 'small'],
                        help='Type of the model to finetune: \'big\' for default ResNet101, \'small\' for ResNet101 with removed layer3')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training (default: 256)')
    parser.add_argument('--checkpoint', type=str, help='Path to the saved teacher\'s weights')
    parser.add_argument('--logging_step', type=int, help='Logging step')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    
    # setting random seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # TODO: data
    train_dataset = CIFAR10(...)
    test_dataset = CIFAR10(...)
    train_loader, test_loader = ...

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.task == "train":
        if args.model_type == "big":
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
            model.fc = torch.nn.Linear(2048, 10)
        elif args.model_type == "small":
            model = resnet101(weights=None)
            model.fc = torch.nn.Linear(2048, 10)
            model.layer3 = torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        else:
            raise Exception("Invalid model type")
        train(model, train_loader, test_loader, device, args.logging_step)
    elif args.task == "distill":
        teacher = resnet101(weights=ResNet101_Weights.DEFAULT)
        teacher.load_state_dict(args.checkpoint)
        teacher.fc = torch.nn.Linear(2048, 10)

        student = resnet101(weights=None)
        student.fc = torch.nn.Linear(2048, 10)
        student.layer3 = torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        distill(student, teacher, train_loader, test_loader, device, args.logging_step)