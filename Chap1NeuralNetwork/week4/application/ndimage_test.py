import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import interpolate
from dnn_app_utils_v3 import *
import matplotlib.image as mpig
import torchvision
from torchvision import transforms

my_image = "1.jpg" # change this to the name of your image file
my_label_y = [0] # the true class of your image (1 -> cat, 0 -> non-cat)
num_px = 64

fname = "images/" + my_image
image = mpig.imread(fname)

image = image/255.
image = transforms.ToTensor()
image = transforms.Resize((64, 64))
torchvision.
image = transforms.ToPILImage(mode=3)
print(image.shape)
plt.imshow(image)
plt.show()

