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
    acc1, acc2, acc3 = accuracies[0], accuracies[1], accuracies[2]
    return torch.abs(acc1 - acc2) < eps and torch.abs(acc2 - acc3) < eps

def train(model,
          train_loader,
          test_loader,
          device,
          checkpoint,
          logging_step=20):
    model = model.to(device)
    # Use the standard Adam optimizer without scheduler.
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
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
            loss = criterion(logits, labels)
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
        test_accuracy = evaluate(model, test_loader, device, step)
        last_accuracies.append(test_accuracy)
        if len(last_accuracies) > 3:
            last_accuracies.pop(0)

    print(f"Finished training after {epoch} epochs!")
    torch.save(model.state_dict(), checkpoint)
  

# evaluation
def evaluate(model,
             test_loader,
             device,
             step):
    model.eval()
    accuracy = 0
    n_batches = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            n_batches += 1
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            prediction = logits.argmax(dim=1)
            accuracy += (prediction == labels).sum() / labels.shape[0]
    wandb.log({"accuracy_test": accuracy / n_batches}, step=step)
    return accuracy / n_batches


# distillation
teacher_layer1_outputs = []
teacher_layer2_outputs = []
teacher_layer4_outputs = []

student_layer1_outputs = []
student_layer2_outputs = []
student_layer4_outputs = []

def hook_fn_teacher(module, input, output):
    teacher_layer1_outputs.append(output[0])
    teacher_layer2_outputs.append(output[1])
    teacher_layer4_outputs.append(output[3])

def hook_fn_student(module, input, output):
    student_layer1_outputs.append(output[0])
    student_layer2_outputs.append(output[1])
    student_layer4_outputs.append(output[3])

def distill(student,
            teacher,
            train_loader,
            test_loader,
            device,
            mse=False,
            logging_step=20):
    criterion = torch.nn.CrossEntropyLoss()
    mse_criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=3e-4)

    teacher, student = teacher.to(device), student.to(device)
    teacher.eval()
    student.train()

    teacher.layer1.register_forward_hook(hook_fn_teacher)
    teacher.layer2.register_forward_hook(hook_fn_teacher)
    teacher.layer4.register_forward_hook(hook_fn_teacher)

    student.layer1.register_forward_hook(hook_fn_student)
    student.layer2.register_forward_hook(hook_fn_student)
    student.layer4.register_forward_hook

    last_accuracies = []
    epoch, step = 0, 0
    current_loss, current_accuracy = 0, 0

    while not stop_criterion(last_accuracies):
        student.train()
        print(f"Epoch: {epoch}")
        for images, labels in tqdm(train_loader):
            optimizer.zero_grad()
            step += 1
            images, labels = images.to(device), labels.to(device)
            logits = student(images)

            with torch.no_grad():
                teacher_logits = teacher(images)

            classification_loss = criterion(logits, labels) # student, classification loss
            # soft-label distillation
            distillation_loss = criterion(
                torch.nn.functional.softmax(logits, dim=1),
                torch.nn.functional.softmax(teacher_logits, dim=1)
            )
            total_loss = distillation_loss + classification_loss

            if mse:
                #  MSE loss between corresponding layer1, layer2 and layer4 features of the student and the teacher
                mse_loss = 0
                mse_loss += mse_criterion(torch.as_tensor(student_layer1_outputs[0]), torch.as_tensor(teacher_layer1_outputs[0]))
                mse_loss += mse_criterion(torch.as_tensor(student_layer2_outputs[0]), torch.as_tensor(teacher_layer2_outputs[0]))
                mse_loss += mse_criterion(torch.as_tensor(student_layer4_outputs[0]), torch.as_tensor(teacher_layer4_outputs[0]))
                total_loss += mse_loss

            # saving my RTX3060 from OOM
            teacher_layer1_outputs.clear()
            teacher_layer2_outputs.clear()
            teacher_layer4_outputs.clear()
            student_layer1_outputs.clear()
            student_layer2_outputs.clear()
            student_layer4_outputs.clear()
            torch.cuda.empty_cache()

            current_loss += total_loss.item() / (logging_step * 3 if mse else logging_step * 2)
            prediction = logits.argmax(dim=1)
            current_accuracy += (prediction == labels).sum() / logging_step / labels.shape[0]

            if step % logging_step == 0:
                wandb.log({"loss_train": current_loss}, step=step)
                wandb.log({"accuracy_train": current_accuracy}, step=step)
                wandb.log({"epoch": epoch}, step=step)
                wandb.log({"distillation_loss": distillation_loss.item() / logging_step}, step=step)
                wandb.log({"clasffication_loss": classification_loss.item() / logging_step}, step=step)
                
                if mse:
                    wandb.log({"mse_loss": mse_loss.item() / logging_step}, step=step)
                current_loss, current_accuracy = 0, 0

            total_loss.backward()
            optimizer.step()

        epoch += 1
        test_accuracy = evaluate(student, test_loader, device, step)
        last_accuracies.append(test_accuracy)
        if len(last_accuracies) > 3:
            last_accuracies.pop(0)

    print(f"Finished training after {epoch} epochs!")
    if mse:
        torch.save(student.state_dict(), "/home/h1de0us/uni/model-compression/task01/checkpoints/student_distill_mse.pt")
    else:
        torch.save(student.state_dict(), "/home/h1de0us/uni/model-compression/task01/checkpoints/student_distill.pt")

