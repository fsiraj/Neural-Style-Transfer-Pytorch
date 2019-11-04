# Imports
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import optim
from torchvision import transforms, models

# CUDA functionality if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_vgg19(style_weights=None):
    '''Initializes pretrained VGG19 feature extractor with parameters frozen
       and style_layers and style_weights defined as attributes.'''

    vgg19 = models.vgg19(pretrained=True).features.to(device)

    # Freezes parameters
    for param in vgg19.parameters():
        param.requires_grad_(False)

    # Style Transfer layers for VGG19
    vgg19.style_layers = {'0' : 'conv1_1',
                          '5' : 'conv2_1',
                          '10': 'conv3_1',
                          '19': 'conv4_1',
                          '21': 'conv4_2',
                          '28': 'conv5_1'
                          }
    # Weights for each style layer
    if style_weights==None:
        vgg19.style_weights = {'conv1_1': 1.0,
                               'conv2_1': 1.0,
                               'conv3_1': 1.0,
                               'conv4_1': 1.0,
                               'conv5_1': 1.0}
    else:
        vgg19.style_weights = {'conv1_1': style_weights[0],
                               'conv2_1': style_weights[1],
                               'conv3_1': style_weights[2],
                               'conv4_1': style_weights[3],
                               'conv5_1': style_weights[4]
                               }

    return vgg19

def load_image(image_path, max_size=400, shape=None):
    '''Loads in image and applies transforms to it'''
    image = Image.open(image_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    transform = transforms.Compose([transforms.Resize(size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))
                                    ])

    image = transform(image)[:3,:,:].unsqueeze(0)

    return image

def content_style_loader(content_path, style_path):
    '''Loads in content_image and style_image from path
       using load_image and displays them'''

    content_image = load_image(content_path).to(device)
    # Style image resized to match content image
    style_image = load_image(style_path, shape=content_image.shape[-2:]).to(device)

    figure, (ax1, ax2) = plt.subplots(1, 2, figsize=[20, 10])
    ax1.imshow(tensor_to_img(content_image))
    ax1.set_title('Content')
    ax2.imshow(tensor_to_img(style_image))
    ax2.set_title('Style')

    return content_image, style_image


def tensor_to_img(tensor):
    '''Converts a torch tensor to a numpy tensor
       to display with matplotlib'''
    # Clone the tensor
    image = tensor.cpu().clone().detach()

    # Remove single dimentsions and reorder channels for matplotlib
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)

    # Un-normalize the image and bring values between 0 and 1
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0,1)

    return image


def feature_extract(image, model, layers):
    '''Passes an image through a model and returns the features
       extracted by the style transfer layers of the model
       in a dictionary'''

    features_dict = {}
    x = image

    for idx, layer in model._modules.items():
        x = layer(x)
        if idx in layers.keys():
            features_dict[layers[idx]] = x

    return features_dict


def gramian_matrix(tensor):
    '''Calculates the gramian matrix of the tensor'''

    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h*w)
    gramian_matrix = torch.mm(tensor, tensor.t())

    return gramian_matrix

def imshow(tensor):
    '''Displays a torch tensor as an image'''

    plt.figure(figsize=[15,15])
    plt.imshow(tensor_to_img(tensor))

def imsave(tensor, path):
    '''Saves a torch tensor as an image specified by path'''
    plt.imsave(path, tensor_to_img(tensor))


# Main function
def transfer_style(content, style, model,
                   iterations=2000, report=500, canvas=None,
                   content_weight=1, style_weight=1e6, lr=0.006):
    '''Returns dictionary with stylized images saved every report iterations
        with multiples of report as indexes.

       Parameters:
       - content, style: Images loaded with load_image.
       - model         : Convolutional Network (VGG19 preferred).
            - model must have attribute:
                style_layers  : Layers from model to use as style feature maps.
                style_weights : Weights for each style_layer.
       - iterations    : Number of iterations to generate the canvas image.
       - report        : Prints out canvas every 'repor' iterations.
       - content_weight: Weight given to content features.
       - style_weight  : Overall weight given to style features. Usually much
                         larger than content_weight.
    '''

    style_layers = model.style_layers
    style_weights = model.style_weights
    # Extracts features from content and style images; generates style gramian matrixes for each layer
    content_features = feature_extract(content, model, style_layers)
    style_features = feature_extract(style, model, style_layers)
    style_grams = {layer: gramian_matrix(style_features[layer]) for layer in style_features}

    # Creates empty dictionary to store canvas iterations in
    canvas_saves = dict()

    # Clones the content image as the starting point of the new image
    if canvas == None:
        canvas = content.clone().requires_grad_(True).to(device)

    optimizer = optim.Adam([canvas], lr=lr)
    start = time.time()
    for i in range(1, iterations+1):

        # Content Loss
        canvas_features = feature_extract(canvas, model, style_layers)
        content_loss = torch.mean((canvas_features['conv4_2'] - content_features['conv4_2'])**2)

        # Style Loss
        style_loss = 0
        for layer in style_weights:

            layer_gram = gramian_matrix(canvas_features[layer])
            style_gram = style_grams[layer]
            _, d, h, w = canvas_features[layer].shape

            style_loss += (torch.mean((layer_gram - style_gram)**2) * style_weights[layer]) / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss

        # Alter canvas
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Iteration count and time
        elapsed = time.time() - start
        elapsed_m, elapsed_s = int(elapsed//60), int(elapsed%60)
        remaining = (elapsed/i)*(iterations-i)
        remaining_m, remaining_s = int(remaining//60), int(remaining%60)

        print(f"Iteration: {i}, ",
              f"Time Elapsed: {elapsed_m:2}m {elapsed_s:2}s, ",
              f"Estimated Time Remaining: {remaining_m:2}m {remaining_s:2}s ",
              f"Iterations/s = {i/elapsed:4.2f}", end="\r")

        if i % report == 0:
            # Display iteration progress
            print(f"Total Loss: {total_loss.item()}")
            plt.imshow(tensor_to_img(canvas))
            plt.show()

            canvas_saves[i] = canvas

    return canvas_saves
