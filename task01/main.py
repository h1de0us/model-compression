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
    parser.add_argument('--logging_step', type=int, help='Logging step', default=20)
    parser.add_argument('--use_mse_during_distillation', type=str, help='Defines whether to use MSE during distillation', default=False)
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

    # data loading
    transforms_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
    test_dataset = CIFAR10(root='./data', train=True, download=True, transform=transforms_cifar)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # running experiment
    wandb.login()
    wandb.init(project="distillation", config=args)

    if args.task == "finetune":
        if args.model_type == "big":
            model = resnet101(weights=ResNet101_Weights.DEFAULT)
            model.fc = torch.nn.Linear(2048, 10)
        elif args.model_type == "small":
            model = resnet101(weights=None)
            model.fc = torch.nn.Linear(2048, 10)
            model.layer3 = torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        else:
            raise Exception("Invalid model type")
        train(model, train_loader, test_loader, device, args.checkpoint, args.logging_step)
    elif args.task == "distill":
        teacher = resnet101(weights=ResNet101_Weights.DEFAULT)
        teacher.fc = torch.nn.Linear(2048, 10)
        teacher.load_state_dict(torch.load(args.checkpoint))
        
        student = resnet101(weights=None)
        student.fc = torch.nn.Linear(2048, 10)
        student.layer3 = torch.nn.Conv2d(512, 1024, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        distill(student, teacher, train_loader, test_loader, device, args.use_mse_during_distillation, args.logging_step)

    wandb.finish()