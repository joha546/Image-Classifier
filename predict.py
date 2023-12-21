import json
import numpy as np
from PIL import Image
import torch
from torch import nn, Tensor
from torchvision import models
import argparse
from workspace_utils import active_session, keep_awake
import sys


def load_categories_to_names(cat_to_name_filepath):

    with open(cat_to_name_filepath, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name


def load_image(image_path):

    loaded_image = Image.open(image_path)

    return loaded_image


def process_image(pil_image):
    # Resize the image
    width, height = pil_image.size
    if width <= height:
        pil_image.thumbnail((256, 256 * (height / width)))
    else:
        pil_image.thumbnail((256 * (width / height), 256))

    # Crop the center of the image
    width, height = pil_image.size
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    cropped_image = pil_image.crop((left, top, right, bottom))

    # Normalize the color channels
    np_image = np.array(cropped_image)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized_image = (np_image - mean) / std

    image_final = normalized_image.transpose((2, 0, 1))

    return image_final


def load_checkpoint(filepath, device):

    if device == 'cuda' and not torch.cuda.is_available():
        print("GPU device is not available. Cannot contiue with processing. "
              "Please check your GPU device, or proceed with prediction on "
              "your local CPU")
        sys.exit()

    try:
        checkpoint = torch.load(filepath, map_location=device)
    except:
        checkpoint = torch.load(filepath)

    # constructing the model
    model = models.vgg11(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Linear(checkpoint['classifier']['input_size'],
                  checkpoint['classifier']['hidden_layers'][0]),
        nn.ReLU(),
        nn.Dropout(p=checkpoint['classifier']['dropout_p']),
        nn.Linear(checkpoint['classifier']['hidden_layers'][0],
                  checkpoint['classifier']['output_size']),
        nn.LogSoftmax(dim=1)
    )

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['model_class_to_idx']

    return model


def predict(image_path, model, classes_to_names, topk=5, device='cpu'):

    loaded_image = load_image(image_path)
    image = process_image(loaded_image)
    image = torch.from_numpy(np.array([image])).float()

    with torch.no_grad():
        image = image.to(device)
        model = model.to(device)
        # idx_to_class = idx_to_class.to(device)

        model.eval()
        output = model.forward(image)

    probs = torch.exp(output)
    top_probs, top_idxs = probs.topk(topk, dim=1)

    class_to_idx = model.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    idxs = Tensor.cpu(top_idxs)[0].numpy()
    top_classes = [idx_to_class[x] for x in idxs]
    top_classes_names = [classes_to_names[c] for c in top_classes]
    # ps = Tensor.cpu(probs).data.numpy().squeeze()

    return top_probs, top_classes_names


def parse_command_line_arguments():

    parser = argparse.ArgumentParser()

    # Positional args
    parser.add_argument('image_filepath', action="store")
    parser.add_argument('model_checkpoint', action="store")

    # Optional args
    parser.add_argument('--category_names', action='store',
                        dest='category_names_filepath',
                        help='Load categories names from given file',
                        default="cat_to_name.json")

    parser.add_argument('--gpu', action='store_true',
                        dest='device',
                        help='Device of prediction processing',
                        default=False)

    parser.add_argument('--top_k', action='store',
                        dest='topk',
                        help='Prediction includes upper topk probabilities',
                        default=5)

    # Parse all args
    results = parser.parse_args()

    return results


if __name__ == "__main__":

    cmd_arguments = parse_command_line_arguments()

    image_filepath = cmd_arguments.image_filepath
    model_checkpoint = cmd_arguments.model_checkpoint
    device = "cuda" if cmd_arguments.device else "cpu"
    category_names_filepath = cmd_arguments.category_names_filepath
    classes_to_names = load_categories_to_names(category_names_filepath)
    topk = int(cmd_arguments.topk)

    model = load_checkpoint(model_checkpoint, device)
    # print(f'Loaded model: \n{model}\n\n')

    print(f'Prediction process started for image: {image_filepath} ....\n')
    top_probs, top_classes_names = predict(image_filepath, model, classes_to_names, topk, device)

    top_probs = Tensor.cpu(top_probs) if device == 'cuda' else top_probs

    print('* Prediction Results:\n\n'
          f'** Image\'s top {topk} predicted probabilities: '
          f'{Tensor.numpy(top_probs)[0]}\n'
          f'** Image\'s top {topk} predicted categories: '
          f'{top_classes_names}\n')