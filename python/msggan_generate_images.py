""" This model was used to generate synthetic images from the trained models.
Make sure to change to the folder that contains the correct model.

 """
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from msggan.models import MsgProGan
from msggan.utils import adjust_color_range

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

# Make sure these hyperparameters are the same as the model was trained with
BATCH_SIZE = 8
DEPTH = 6
IMG_HEIGHT = 2**(DEPTH + 1)
IMG_WIDTH = 2**(DEPTH + 1)
CHANNELS = 3

# Create an identical model
model = MsgProGan(channels=CHANNELS,
                  folder="python/saved_models/msggan_nemo_sediment", use_eql=True, depth=DEPTH)

# Load weights
model.load_model()

# Generate images...
model.generate_images(50000)
