""" This module contains some utility functions that perform miscellaneous tasks. """
import os
import sys 

import numpy as np
import tensorflow as tf

from tqdm import tqdm

def adjust_color_range(image):
    """ Adjust colors by chaging pixel value range from [-1, 1] to [0, 1]. """
    image = image * 0.5 + 0.5
    
    return tf.clip_by_value(image, 0, 1)


def create_path(path_name):
    if not os.path.exists(path_name):
        os.makedirs(f"{path_name}")


def pbar(total_imgs, epoch, epochs):
    bar = tqdm(total=(total_imgs),
               file=sys.stdout,
            #    leave=False,
               #    ncols=int(get_terminal_width() * .9),
               desc=tqdm.write(f'Training epoch {epoch+1} to {epochs}'),
               #    postfix={
               #        'g_loss': f'{0:6.3f}',
               #        'd_loss': f'{0:6.3f}',
               #        1: 1
               #    },
               #    bar_format='{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  '
               #    'ETA: {remaining}  Elapsed Time: {elapsed}  '
               #    'G Loss: {postfix[g_loss]}  D Loss: {postfix['
               #    'd_loss]}',
               unit=' images',
               miniters=1)
    return bar

