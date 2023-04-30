"""
Use the following augmentation methods on the sample image under data/sample.png
and save the result under this path: 'data/sample_augmented.png'

Note:
    - use torchvision.transforms
    - use the following augmentation methods with the same order as below:
        * affine: degrees: ±5, 
                  translation= 0.1 of width and height, 
                  scale: 0.9-1.1 of the original size
        * rotation ±5 degrees,
        * horizontal flip with a probablity of 0.5
        * center crop with height=320 and width=640
        * resize to height=160 and width=320
        * color jitter with:  brightness=0.5, 
                              contrast=0.5, 
                              saturation=0.4, 
                              hue=0.2
    - use default values for anything unspecified
"""

import torch
from torchvision import transforms as T
import numpy as np
import cv2

torch.manual_seed(8)
np.random.seed(8)

img = cv2.imread('data/sample.png')

# comment out: visualize the original image
# cv2.imshow("image", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

transform = T.Compose([
    T.ToPILImage(), # convert to PIL image to ensure the following transforms work
    T.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    T.RandomRotation(degrees=5),
    T.RandomHorizontalFlip(p=0.5),
    T.CenterCrop((320, 640)),
    T.Resize((160, 320)),
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.2),
    T.ToTensor()
])

file_name = 'data/sample_augmented.png'
img_augmented = transform(img) # img_augmented: torch.Size([3, 160, 320])
img_augmented = img_augmented.permute(1, 2, 0).numpy() # img_augmented: (160, 320, 3)
img_augmented = img_augmented * 255
cv2.imwrite('data/sample_augmented.png',img_augmented)

# comment out: visualize the augmented image
# img_aug = cv2.imread('data/sample_augmented.png')
# cv2.imshow("image_aug", img_aug)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

