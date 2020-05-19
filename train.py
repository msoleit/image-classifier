import argparse
import os
import torch
from torch import optim
from torch import nn
import utils
from model import setup_model, train

parser = argparse.ArgumentParser(description='train a pre-trained neural network model on given training data.')
parser.add_argument('data_dir', type=str,
                    help='directory of the training data')
parser.add_argument('--save_dir', dest='save_dir', type=str,
                    help='Set directory to save checkpoints', default=os.getcwd())
parser.add_argument('--arch', dest='architecture', type=str,
                    help='Choose architecture: vgg16, vgg11, resnet18', default='vgg16')
parser.add_argument('--learning_rate', dest='learning_rate', type=float,
                    help='Set learning rate', default=0.003)
parser.add_argument('--hidden_units', dest='hidden_units', type=int,
                    help='Set number of hidden units', default=512)
parser.add_argument('--epochs', dest='epochs', type=int,
                    help='Set number of epochs', default=10)
parser.add_argument('--gpu', action='store_true',default=False, dest='gpu',
                    help='Use gpu for training')

args = parser.parse_args()

data_dir = args.data_dir
architecture = args.architecture
hidden_units = args.hidden_units
learning_rate = args.learning_rate
epochs = args.epochs

if not os.path.isdir(data_dir):
    print('{} is not a valid directory'.format(data_dir))
    exit()
if not utils.is_valid_architecture(architecture):
    print('{} is not a valid architecture'.format(architecture))
    exit()
if not torch.cuda.is_available() and args.gpu:
    print('WARNING : No Cuda available for training, will use CPU')

#Load data
trainloader, validloader, testloader = utils.load_data(data_dir)

# Get torchvision architecture
pre_trained_model = utils.get_torchvision_model(architecture)

# Build network
in_features = utils.get_input_features(architecture)
model = setup_model(pre_trained_model, in_features, hidden_units)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

# Train the network
train(model, trainloader, validloader, criterion, optimizer, args.gpu, epochs)

# Save the model checkpoint
class_to_idx = utils.get_datasets(data_dir)[0].class_to_idx
save_path = args.save_dir + '/checkpoint.pth'
utils.save_checkpoint(model, optimizer, architecture, hidden_units, learning_rate , epochs, class_to_idx, save_path)