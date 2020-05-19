import torch
from torchvision import models, datasets, transforms
from model import setup_model
from PIL import Image
import numpy as np

def is_valid_architecture(arch):
    valid_architecture = ['vgg11', 'vgg16', 'alexnet']
    return arch in valid_architecture

def get_torchvision_model(arch):
    if arch == 'vgg11':
        return models.vgg11(pretrained=True)
    elif arch == 'vgg16':
        return models.vgg16(pretrained=True)
    elif arch == 'alexnet':
        return models.alexnet(pretrained=True)
    else:
        raise Exception('{} is not supported architecture'.format(arch))

def get_input_features(arch):
    in_features = {'vgg11' : 25088,
                   'vgg16' : 25088,
                   'alexnet': 9216
                   }
    return in_features[arch]

def get_datasets(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean,std)])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean,std)])
                                            
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return train_data, valid_data, test_data

def load_data(data_dir):
    train_data, valid_data, test_data = get_datasets(data_dir)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return trainloader, validloader, testloader

def save_checkpoint(model, optimizer, arch, hidden_units, learning_rate, epochs, class_to_idx, save_path, dropout=0.05):
    model.to('cpu')
    checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': class_to_idx,
                  'optimizer_state': optimizer.state_dict(),
                  'arch': arch,
                  'hidden_units': hidden_units,
                  'dropout': dropout,
                  'learning_rate': learning_rate,
                  'num_epochs': epochs
              }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    arch = checkpoint['arch']
    in_features = get_input_features(arch)
    pretrained_model = get_torchvision_model(arch)
    hidden_units = checkpoint['hidden_units']
    model = setup_model(pretrained_model, in_features, hidden_units)
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # open
    image = Image.open(image_path)

    # resize
    image = image.resize((256,256))
    
    # crop
    width, height = image.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    image = image.crop((left, top, right, bottom))
    
    # normalize
    np_image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # transpose
    np_image = np_image.transpose((2,1,0))
    
    return torch.from_numpy(np_image).type(torch.FloatTensor)

def convert_classes_to_names(classes, cat_to_name):
    ''' Converts classes to category names using cat_to_name dict,
        returns category names
    '''
    cat_names = []
    for cls in classes:
        cat_names.append(cat_to_name[str(cls)])
    return cat_names