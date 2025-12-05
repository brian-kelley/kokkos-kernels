# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import sys
from pathlib import Path
import math
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
from time import perf_counter

sys.path.append(str(Path(__file__).absolute().parent))
#from utils._example_utils import (
#    top3_possibilities,
#    load_labels
#)

def load_and_preprocess_image_file(path: str):
    img = Image.open(path).convert("RGB")
    # preprocessing pipeline
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img_preprocessed = preprocess(img)
    return torch.unsqueeze(img_preprocessed, 0)

batch = 64

img = load_and_preprocess_image_file("dog.jpg")
print("Image shape before bcast:")
print(img.shape)
img = torch.broadcast_to(img, (batch, 3, 224, 224))
print("Image shape after bcast:")
print(img.shape)

print("Cuda available? ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU info:", torch.cuda.get_device_name(0))

img = img + 1e-4 * torch.rand((batch, 3, 224, 224), dtype=torch.float)

if torch.cuda.is_available():
    img = img.to("cuda")

resnet18 = models.resnet18(pretrained=True).to("cuda").eval()

# Warmup
resultWarmup = resnet18.forward(img)

print("Warmup result shape:", resultWarmup.shape)

#torch.backends.cudnn.enabled = False

trials = 1000
time_start = perf_counter()

for i in range(trials):
    result = resnet18.forward(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

time_end = perf_counter()
print("torch interp, averaged over", trials, "trials:", (time_end - time_start) / trials)

resnet18_compiled = torch.compile(resnet18)
resultWarmup = resnet18_compiled.forward(img)
print("Warmup result shape:", resultWarmup.shape)

time_start = perf_counter()

for i in range(trials):
    result = resnet18_compiled.forward(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

time_end = perf_counter()

print("torch.compile, averaged over", trials, "trials:", (time_end - time_start) / trials)
