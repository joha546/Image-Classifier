import time
import json
import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from workspace_utils import active_session
import argparse


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms_training = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms_validation = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    data_transforms_test = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image_datasets_training = datasets.ImageFolder(train_dir, transform=data_transforms_training)
    image_datasets_validation = datasets.ImageFolder(valid_dir, transform=data_transforms_validation)
    image_datasets_test = datasets.ImageFolder(test_dir, transform=data_transforms_test)

    dataloaders_training = torch.utils.data.DataLoader(image_datasets_training, shuffle=True, batch_size=128)
    dataloaders_validation = torch.utils.data.DataLoader(image_datasets_validation, shuffle=True, batch_size=128)
    dataloaders_test = torch.utils.data.DataLoader(image_datasets_test, shuffle=True, batch_size=128)

    return {"training_dataloader": dataloaders_training,
            "validation_dataloader": dataloaders_validation,
            "testing_dataloader": dataloaders_test,
            "class_to_idx": image_datasets_training.class_to_idx}


def load_categories_to_names(cat_to_name_filepath):
    with open(cat_to_name_filepath, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def build_and_train_model(dataloaders_training, dataloaders_validation,
                          class_to_idx, learning_rate=0.001, epochs=5,
                          hidden_units=512, arch='vgg11', device='cpu'):

    model = getattr(models, arch)(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    with active_session():
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

        model.to(device)

        steps = 0
        train_losses, test_losses = [], []


        for e in range(epochs):
            running_loss = 0

            for ii, (images, labels) in enumerate(dataloaders_training):
                # TRAINING happens here:

                # setup
                images, labels = images.to(device), labels.to(device)
                start = time.time()
                optimizer.zero_grad()

                # feed forward
                outputs = model.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # record loss
                running_loss += loss.item()

            else:
                test_loss = 0
                accuracy = 0

                with torch.no_grad():
                    model.eval()

                    for images, labels in dataloaders_validation:
                        images, labels = images.to(device), labels.to(device)

                        outputs = model.forward(images)
                        loss = criterion(outputs, labels)

                        test_loss += loss.item()

                        probs = torch.exp(outputs)
                        top_probs, top_class = probs.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        if device == 'cuda':
                            accuracy += torch.mean(
                                equals.type(torch.cuda.FloatTensor))
                        else:
                            accuracy += torch.mean(
                                equals.type(torch.FloatTensor))

                model.train()

                train_losses.append(running_loss / len(dataloaders_training))
                test_losses.append(test_loss / len(dataloaders_validation))

                print(
                    f'epoch {e + 1}/{epochs}',
                    f'Training Loss: {running_loss/len(dataloaders_training)}',
                    f'Test Loss: {test_loss / len(dataloaders_validation)}',
                    f'Accuracy: {accuracy / len(dataloaders_validation)}')

    model.class_to_idx = class_to_idx

    return {"model": model, "criterion": criterion, "optimizer": optimizer}


def test_trained_model(model, criterion, dataloaders_test, device):
    with active_session():
        model.to(device)

        test_losses = []

        with torch.no_grad():
            model.eval()

            equals_sum = 0
            items_count = 0

            for images, labels in dataloaders_test:
                # Testing happens here:
                test_loss = 0
                accuracy = 0

                images, labels = images.to(device), labels.to(device)

                outputs = model.forward(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

                probs = torch.exp(outputs)
                top_probs, top_class = probs.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                items_count += len(equals)

                if device == 'cuda':
                    equals_sum += torch.sum(equals.type(torch.cuda.FloatTensor))
                else:
                    equals_sum += torch.sum(equals.type(torch.FloatTensor))

            test_losses.append(test_loss / len(dataloaders_test))

            print(f'Test Loss: {test_loss / len(dataloaders_test)}',
                  f'Accuracy: {equals_sum / items_count}')

    return None


def save_checkpoint(model, optimizer, save_directory, lr=0.001, epochs=5,
                    hidden_units=512, device='cpu'):

    checkpoint = {
        'classifier': {
            'input_size': 25088,
            'output_size': 102,
            'hidden_layers': [hidden_units],
            'dropout_p': 0.2
        },
        'state_dict': model.state_dict(),
        'model_class_to_idx': model.class_to_idx,
        'training': {
            'optimizer': optimizer.state_dict,
            'epochs': epochs,
            'lr': lr
        }
    }

    torch.save(checkpoint, save_directory + '/checkpoint_' + device + '.pth')
    return None


def parse_command_line_arguments():

    parser = argparse.ArgumentParser()

    # Positional args
    parser.add_argument('data_directory', action="store")

    # Optional args
    parser.add_argument('--save_dir', action='store',
                        dest='save_dir',
                        help='Load categories names from given file',
                        default="checkpoint.pth")

    parser.add_argument('--gpu', action='store_true',
                        dest='device',
                        help='Device of prediction processing',
                        default=False)

    parser.add_argument('--arch', action='store',
                        dest='arch',
                        help='Name of pre-trained network used for training',
                        default="vgg11")

    parser.add_argument('--learning_rate', action='store',
                        dest='learning_rate',
                        help='value of training learning rate',
                        default=0.001)

    parser.add_argument('--hidden_units', action='store',
                        dest='hidden_units',
                        help='Number of units in the fully-connected hidden '
                             'layer of the neural netwrork',
                        default=512)

    parser.add_argument('--epochs', action='store',
                        dest='epochs',
                        help='Number of training epochs',
                        default=5)

    # Parse all args
    results = parser.parse_args()

    return results


if __name__ == "__main__":
    cmd_arguments = parse_command_line_arguments()

    data_directory = cmd_arguments.data_directory
    save_dir = cmd_arguments.save_dir
    device = "cuda" if cmd_arguments.device else "cpu"
    arch = cmd_arguments.arch
    learning_rate = float(cmd_arguments.learning_rate)
    hidden_units = int(cmd_arguments.hidden_units)
    epochs = int(cmd_arguments.epochs)

    # load data
    print('* Loading data in progress ...')
    dataloaders = load_data(data_directory)
    training_dataloader = dataloaders["training_dataloader"]
    validation_dataloader = dataloaders["validation_dataloader"]
    testing_dataloader = dataloaders["testing_dataloader"]
    class_to_idx = dataloaders["class_to_idx"]
    print('* Data loaded successfully!\n')

    # start training and validation:
    print('* Building and training model in progress ...')
    print('* Following are training loss, validation loss, and model accuracy:\n')
    model_details = build_and_train_model(
        training_dataloader, validation_dataloader, class_to_idx,
        learning_rate, epochs, hidden_units, arch, device)

    model = model_details['model']
    criterion = model_details['criterion']
    optimizer = model_details['optimizer']
    print('\n* Finished training model successfully!\n')
    print(f'--> This is our trained model:\n\n{model}\n\n')

    # test model
    print('* Let\'s test our model against testing data ...\n')
    test_trained_model(model, criterion, testing_dataloader, device)
    print('\n* Done testing successfully!\n')

    # save checkpoint
    print(f'* Saving model checkpoint as {save_dir}/checkpoint_{device}.pth')
    save_checkpoint(model, optimizer, save_dir, learning_rate, epochs,
                    hidden_units, device)
    print('* Saved checkpoint successfully!\n')