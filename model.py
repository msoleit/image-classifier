import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def setup_model(model, in_features, hidden_units, drop_p=0.05):
    ''' Builds a feedforward network from a given pre-trained model with arbitrary hidden layers.
        
        Arguments
        ---------
        model: torchvision model
        hidden_units: integer, the size of the hidden layer
    '''
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(nn.Linear(in_features, hidden_units),
                 nn.ReLU(),
                 nn.Dropout(p=drop_p),
                 nn.Linear(hidden_units,102),
                 nn.LogSoftmax(dim=1)
                )
    model.classifier = classifier
    
    return model

def validation(model, device, validloader, criterion):
    valid_loss = 0
    accuracy = 0
        
    model.to(device)
    for inputs, labels in validloader:
        inputs, labels = inputs.to(device), labels.to(device)
        log_ps = model(inputs)
        batch_loss = criterion(log_ps, labels)
                    
        valid_loss += batch_loss.item()
                    
        #accuracy
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return valid_loss, accuracy


def train(model, trainloader, validloader, criterion, optimizer,gpu=False, epochs=5, print_every=40):
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')
    model.to(device)
    steps = 0
    running_loss = 0
    print('...Starting training...')
    for e in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            log_ps = model(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if steps % print_every == 0:
                model.eval()
                
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, device, validloader, criterion)
                 
                print(f"Epoch {e+1}/{epochs}..."
                      f"Train Loss: {running_loss/print_every:.3f}..."
                      f"Validation Loss: {valid_loss/len(validloader):.3f}..."
                      f"Validation Accuracy: {accuracy/len(validloader):.3f}")
            
                running_loss = 0
                model.train()
    print("..Training Done.."
          f"{steps} steps taken")

def predict(image, model, topk=1, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')

    image = image.unsqueeze_(0)
    image = image.to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        log_p = model(image)
        p = torch.exp(log_p)
        top_p, top_classes = p.topk(topk, dim=1)
        top_p = np.array(top_p)
        top_classes = np.array(top_classes)
        idx_to_class = {value:key for key, value in model.class_to_idx.items()}
        
        probs = [prob for prob in top_p[0]]
        classes = [idx_to_class[cls] for cls in top_classes[0]]
        
        return probs, classes