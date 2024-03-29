import torch
import wandb
from tqdm import tqdm

def stop_criterion(accuracies, eps=0.01):
    '''
    Training stopping criterion: accuracy (measured from 0 to 1) stabilizes in the second digit after decimal during at least 2 epochs on test set. 
    That means that you must satisfy condition torch.abs(acc - acc_prev) < 0.01 
    for at least two epochs in a row.
    '''
    if len(accuracies) < 3: return False
    acc1, acc2, acc3 = accuracies
    return torch.abs(acc1 - acc2) < eps and torch.abs(acc2 - acc3) < eps

def train(model,
          train_loader,
          test_loader,
          device,
          logging_step=20):
    model = model.to(device)
    # Use the standard Adam optimizer without scheduler.
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()

    last_accuracies = []
    epoch, step = 0, 0
    current_loss, current_accuracy = 0, 0
    # training
    while not stop_criterion(last_accuracies):
        model.train()
        print(f"Epoch: {epoch}")
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            step += 1
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels).item()
            current_loss += loss.item() / logging_step
            prediction = logits.argmax(dim=1)
            current_accuracy += (prediction == labels).sum() / logging_step / labels.shape[0]

            if step % logging_step == 0:
                wandb.log({"loss_train": current_loss}, step=step)
                wandb.log({"accuracy_train": current_accuracy}, step=step)
                wandb.log({"epoch": epoch}, step=step)
                current_loss, current_accuracy = 0, 0

            loss.backward()
            optimizer.step()

        epoch += 1
        test_accuracy = evaluate(model, test_loader, step)
        last_accuracies.append(test_accuracy)
        if len(last_accuracies) > 3:
            last_accuracies.pop(0)

    print(f"Finished training after {epoch} epochs!")
  

# evaluation
def evaluate(model,
             test_loader,
             device,
             step):
    model.eval()
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        prediction = logits.argmax(dim=1)
        accuracy += (prediction == labels).sum() / labels.shape[0]
        wandb.log({"accuracy_test": accuracy}, step=step)


# distillation
def distill(student,
            teacher,
            train_loader,
            test_loader,
            device,
            logging_step=20):
    pass


