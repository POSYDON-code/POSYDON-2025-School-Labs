"""

2025-09-22
Description: this file implements utility functions to be used in the summer school tutorials
pertaining to machine learning components of POSYDON.
Author(s): Philipp M. Srivastava

"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def show_figure(figure_path, source):

    img = Image.open(figure_path)
    img = np.array(img)
    plt.imshow(img)
    plt.axis("off")

    print(f"Source: {source}")