import sys
from typing import Union, Callable, Tuple
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from multiprocessing import cpu_count
from collections import OrderedDict
import matplotlib.pyplot as plt
import torch
from PIL import Image
import io
import requests
import tarfile
import time

transform_pipeline = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def download_dataset():
    # Download of the dataset
    DATASET_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
    target_path ="/content/Imagenette.tgz"
    download(DATASET_URL, target_path)

    # Unzip the dataset
    tar = tarfile.open("/content/Imagenette.tgz")
    tar.extractall()
    tar.close()

    DICT_URL = "https://raw.githubusercontent.com/val-iisc/nag/master/misc/ilsvrc_synsets.txt"

    # Download the file
    response = requests.get(DICT_URL)
    r = response.content.decode('utf-8')

    # We must realise some preprocessing as the file is a simple .txt files
    list_classes = r.split("\n")
    # We must delete the last element which empty -> ''
    list_classes.pop(-1)

    dictionary = pd.DataFrame(columns=['key','object'])
    for line in list_classes:
    key, obj = line.split(" ",1)
    objects = [o for o in obj.split(", ")]
    dictionary = dictionary.append({'key':key, 'object':objects}, ignore_index=True)

# This is the function used to download the file while displaying a progress bar (not really necessary)
def download(url, filename):
    with open(filename, 'wb') as f:
        response = requests.get(url, stream=True, allow_redirects=True)
        total = response.headers.get('content-length')

        if total is None:
            f.write(response.content)
        else:
            downloaded = 0
            total = int(total)
            for data in response.iter_content(chunk_size=max(int(total/1000), 1024*1024)):
                downloaded += len(data)
                f.write(data)
                done = int(50*downloaded/total)
                sys.stdout.write('\r[{}{}]'.format('█' * done, '.' * (50-done)))
                sys.stdout.flush()
    sys.stdout.write('\n')

def load_images():

    data_dir = "/content/imagenette2-160/train"

    # An object containing all the images of the image folder
    # We pass the pipeline as a parameter of the function as it uses it to transform the images
    image_dataset = datasets.ImageFolder(data_dir, transform = transform_pipeline)

    imagenette_classes_names = image_dataset.classes
    imagenet_classes_names = [dictionary[dictionary.key == k].index[0] for k in imagenette_classes_names]
    image_dataset.target_transform = lambda id: dictionary[dictionary.key == image_dataset.classes[id]].index[0]

    # An iterator object that allows to work with batches instead of the whole dataset at once
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=10, shuffle=True)

    real_classes_names = [dictionary.iloc[k].object[0] for k in imagenet_classes_names]

    # Get size and class_names for further use
    dataset_size = len(image_dataset)

    # Print some information about the dataset
    print('There are {} images in the dataset \nIt is split in 12 different classes. \n\nThese classes are : \n{}. \n\nThanks to the dictonary we retreive the real classes names : \n{}'.format(dataset_size, imagenette_classes_names, real_classes_names))

    return image_dataset, dataloader

def load_HQ_images():
    cat = "https://icatcare.org/app/uploads/2018/07/Thinking-of-getting-a-cat.png"
    crocodile = "https://cdn.britannica.com/84/198884-050-A37B8971/crocodile-Nile-swath-one-sub-Saharan-Africa-Madagascar.jpg"
    tractor = "https://www.ft.com/__origami/service/image/v2/images/raw/https%3A%2F%2Fs3-ap-northeast-1.amazonaws.com%2Fpsh-ex-ftnikkei-3937bb4%2Fimages%2F3%2F7%2F0%2F4%2F1444073-11-eng-GB%2F20171003_Mahindra.jpg?source=nar-cms"

    response = requests.get(cat)
    image_cat = Image.open(io.BytesIO(response.content))
    image_cat = transform_pipeline(image_cat)

    response = requests.get(crocodile)
    image_crocodile = Image.open(io.BytesIO(response.content))
    image_crocodile = transform_pipeline(image_crocodile)

    response = requests.get(tractor)
    image_tractor = Image.open(io.BytesIO(response.content))
    image_tractor = transform_pipeline(image_tractor)

    images = torch.stack([image_cat,image_crocodile,image_tractor])
    labels = torch.Tensor([281,49,866]).int()
    images_HQ = [images, labels]

    display_images(images,labels.numpy())

  return images_HQ


def display_images(images, labels, pred_lab = None, dict_lab=1):
  
  # Set the size of the figure depending on the number of images to display
  nb_images = len(images)

  heigt, width = 12, (4*nb_images/4)
  fig = plt.figure(figsize=(heigt,width))

  # We reverse the transformation pipeline in order to display the images
  for i in range(nb_images):
    img = images[i]
    img = img.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)

  # We create the image grid 
    plt.subplot(int(nb_images/4)+1,4,i+1, frame_on = False, yticks=[], xticks=[])
    plt.imshow(img)

  # We display the  |T : True      | labels pairs above each image
  #                 |P : Predicted |
  # We only display the first definition of the 
  # class so it is not too long and the titles don't overlap
    label = dictionary.iloc[labels[i]].object[0] if (dict_lab==1) else labels[i]
    if pred_lab == None :
      plt.title("T: {}".format(label))
    else :
      predicted = dictionary.iloc[pred_lab[i]].object[0]
      plt.title("T: {}\nP: {}"
      .format(label, predicted))
    plt.show()

def unnormalise(x):
    y = x
    for i, (mean, std) in enumerate(zip([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])):
        y[i, :, :] = x[i, :, :]*std + mean
        
    return y

# Get image
def adversarial_example_pgd(model, nb_images, images, labels, targeted=None, nb_iter=200, step_size=0.05, epsilon=1):
  X, Y = images.clone(), labels
  if nb_images > len(images) : nb_images = 1
  type_attack = "Targeted" if not targeted == None else "Untargeted"
  if type_attack == "Untargeted" :
    print(f"Performing PGD Untargeted Attack, with {nb_iter} iterations with a step of {step_size} and with a perturbation threshold of {epsilon}")
  else : 
    print(f"Performing a Targeted Attack targeting the class No {targeted}, with {nb_iter} iterations with a step of {step_size} and with a perturbation threshold of {epsilon}")
  adv_images = []
  adv_predictions = []
  for i in range(nb_images):
    x = X[i]
    y = Y[i]
    x_plot = unnormalise(x).permute(1,2,0).numpy()
    x = x.unsqueeze(0).to(DEVICE)

    # Get regular prediction
    y_pred = model(x)
    pred_class = y_pred.cpu().data.numpy().argmax()

    # Performing Projected Gradient Descent attack
    y_target = torch.Tensor([targeted]).to(DEVICE).long() if not targeted == None else None
    x_adv = pgd(model, x, torch.Tensor([y]).to(DEVICE).long(), torch.nn.CrossEntropyLoss(),
                k=nb_iter, step=step_size, eps=epsilon, norm=2,
                y_target=y_target
              )
    adv_images.append(x_adv.cpu())
    y_pred_adv = model(x_adv)
    pred_class_adv = y_pred_adv.cpu().data.numpy().argmax()
    adv_predictions.append(pred_class_adv)

    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle(f'{type_attack} Attack ')
    
    axes[0].imshow(x_plot)
    axes[0].set_title(
        f'Real name = {dictionary.iloc[y.item()].object[0]} ({y.item()})\n'
        f'Predicted name = {dictionary.iloc[pred_class.item()].object[0]}\n'
        f'P({pred_class.item()}) = '
        f'{np.round(y_pred.softmax(dim=1)[0, pred_class].item(), 2)}'
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Adversarial perturbation scaled for increased visibility
    axes[1].imshow(unnormalise(50*(x_adv - x).squeeze(0)).permute(1, 2, 0).cpu().numpy())
    axes[1].set_title(
        f'Noise added to the original image with epsilon = {epsilon}\nand {nb_iter} iterations with a step of {step_size}'
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[2].set_title(
        f'Predicted name = {dictionary.iloc[pred_class_adv.item()].object[0]}\n'
        f'P({pred_class_adv.item()}) = {y_pred_adv.softmax(dim=1)[0, pred_class_adv].item()}'
    )
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.show()
    print(adv_images.type)
    dataloader_adv = [torch.stack(adv_images), torch.Tensor(adv_predictions)]
  return dataloader_adv

# Get image
def adversarial_example_fgsm(model, nb_images, images, labels, targeted=None, nb_iter=60, step_size=0.01, epsilon=0.5):
  X, Y = images.clone(), labels
  if nb_images > len(images) : nb_images = 1
  type_attack = "Targeted" if not targeted == None else "Untargeted"
  if type_attack == "Untargeted" :
    print(f"Performing FGSM Untargeted Attack, with {nb_iter} iterations with a step of {step_size} and with a perturbation threshold of {epsilon}")
  else : 
    print(f"Performing FGSM Targeted Attack targeting the class No {targeted}, with {nb_iter} iterations with a step of {step_size} and with a perturbation threshold of {epsilon}")

  for i in range(nb_images):
    x = X[i]
    y = Y[i]
    x_plot = unnormalise(x).permute(1,2,0).numpy()
    x = x.unsqueeze(0).to(DEVICE)

    # Get regular prediction
    y_pred = model(x)
    pred_class = y_pred.cpu().data.numpy().argmax()

    # Performing Iterated FGSM attack 
    y_target = torch.Tensor([targeted]).to(DEVICE).long() if not targeted == None else None
    x_adv = iterated_fgsm(model, x, torch.Tensor([y]).to(DEVICE).long(), torch.nn.CrossEntropyLoss(),
                          k=nb_iter, step=step_size, eps=epsilon, norm='inf',
                          y_target=y_target
                          )


    y_pred_adv = model(x_adv)
    pred_class_adv = y_pred_adv.cpu().data.numpy().argmax()

    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    fig.suptitle(f'{type_attack} Attack ')
    
    axes[0].imshow(x_plot)
    axes[0].set_title(
        f'Real name = {dictionary.iloc[y.item()].object[0]} ({y.item()})\n'
        f'Predicted name = {dictionary.iloc[pred_class.item()].object[0]}\n'
        f'P({pred_class.item()}) = '
        f'{np.round(y_pred.softmax(dim=1)[0, pred_class].item(), 2)}'
    )
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Adversarial perturbation scaled for increased visibility
    axes[1].imshow(unnormalise(50*(x_adv - x).squeeze(0)).permute(1, 2, 0).cpu().numpy())
    axes[1].set_title(
        f'Noise added to the original image with epsilon = {epsilon}\nand {nb_iter} iterations with a step of {step_size}'
    )
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(x_adv.squeeze(0).permute(1, 2, 0).cpu().numpy())
    axes[2].set_title(
        f'Predicted name = {dictionary.iloc[pred_class_adv.item()].object[0]}\n'
        f'P({pred_class_adv.item()}) = {y_pred_adv.softmax(dim=1)[0, pred_class_adv].item()}'
    )
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    plt.show()