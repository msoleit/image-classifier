import argparse
import utils
from model import predict
import torch
import json

parser = argparse.ArgumentParser(description='predict flower name from an image along with the probability of that name.')
parser.add_argument('path_to_image', type=str,
                    help='Path of the image to predict')
parser.add_argument('checkpoint', type=str,
                    help='Model checkpoint to use for prediction')
parser.add_argument('--top_k', dest='top_k', type=int,
                    help='Return top K most likely class', default=1)
parser.add_argument('--category_names', dest='cat_names', type=str,
                    help='Use a mapping of categories to real names', default=None)
parser.add_argument('--gpu', action='store_true',default=False, dest='gpu',
                    help='Use GPU for inference')

args = parser.parse_args()

checkpoint = args.checkpoint
image_path = args.path_to_image
top_k = args.top_k
gpu = args.gpu
cat_names_file = args.cat_names

# Load Checkpoint
model = utils.load_checkpoint(checkpoint)

# Process Image
image = utils.process_image(image_path)

# Predict
if not torch.cuda.is_available() and gpu:
   print('WARNING : No Cuda available for inference, will use CPU')
probs, classes = predict(image, model, top_k, gpu)
if not cat_names_file is None:
    with open(cat_names_file, 'r') as f:
         cat_to_name = json.load(f)
    classes = utils.convert_classes_to_names(classes, cat_to_name)
print(f"Top {top_k} predicted classes :")
print("Class   :  Probability")
print("--------------------------")
for idx in range(len(classes)):
    print(f"{classes[idx]} : {probs[idx]}")