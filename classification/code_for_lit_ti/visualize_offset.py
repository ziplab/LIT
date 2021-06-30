import os
import sys
import pprint
# import cv2
from torchvision import transforms
import utils
from params import args
import torch
import matplotlib.pyplot as plt
import lit
import matplotlib.patches as patches
from PIL import Image
import cv2
import numpy as np
from show_offset import show_dconv_offset

if __name__ == '__main__':
    from timm.models import create_model
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
    )

    checkpoint = torch.load(args.resume, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.cuda()

    if args.vis_image is None:
        print('Please provide a targe image to be visualized')
        sys.exit(0)

    images = [args.vis_image]
    data = []
    image_all = []
    for image_path in images:
        img = cv2.imread(image_path)
        cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform(cv_img)
        data.append(img)
        store_image = img.permute(1,2,0).numpy()
        image_all.append(store_image)

    data = torch.stack(data).cuda()
    with torch.no_grad():
        all_offsets = model.get_all_offsets(data)

    for idx, image in enumerate(image_all):
        image_name = images[idx].split('/')[-1].split('.')[0]
        check_offsets = []
        for off in all_offsets:
            c_off = off[idx]
            c,w,h = c_off.shape
            c_off = c_off.reshape(4, 2, w, h)
            check_offsets.append(c_off)
        check_offsets.reverse()
        show_dconv_offset(image, check_offsets, filter_size=2, dilation=1, pad=0, plot_area=1, plot_level=3, step=[1,1], image_name=image_name)



