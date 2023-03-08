import time
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from bing_image_downloader import downloader
import time
import os
import copy

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


downloader.download(
    "hammer",
    limit=100,
    output_dir="data",
    adult_filter_off=True,
    force_replace=False,
    timeout=60,
)

