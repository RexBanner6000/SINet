import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
from scipy import misc  # NOTES: pip install scipy == 1.2.2 (prerequisite!)
from Src.SINet import SINet_ResNet50
from Src.utils.Dataloader import test_dataset
from Src.utils.trainer import eval_mae, numpy2tensor
from PIL import Image
import torchvision.transforms as transforms
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='the snapshot input size')
parser.add_argument('--model_path', type=str,
                    default='./SINet_40.pth')
parser.add_argument('--output_dir', type=str,
                    default='./Result/')
parser.add_argument('--input_dir', type=str, default=r'D:\spectral_design\images\defence_imagery')
opt = parser.parse_args()

model = SINet_ResNet50().cuda()
model.load_state_dict(torch.load(opt.model_path))
model.eval()

transform = transforms.Compose([
            transforms.Resize((opt.testsize, opt.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

os.makedirs(opt.output_dir, exist_ok=True)
img_files = os.listdir(opt.input_dir)
for img_file in img_files:
    print(img_file)
    img = Image.open(f"{opt.input_dir}/{img_file}").convert('RGB')
    cv_img = cv2.imread(f"{opt.input_dir}/{img_file}")
    image = transform(img).unsqueeze(0).cuda()

    _, cam = model(image)
    cam = F.upsample(cam, size=(img.height, img.width), mode='bilinear', align_corners=True)
    cam = cam.sigmoid().data.cpu().numpy().squeeze()
    # normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heatmap = cv2.applyColorMap((cam * 255).astype('uint8'), cv2.COLORMAP_JET)
    fin = cv2.addWeighted(heatmap, 0.3, cv_img, 0.7, 0)
    # misc.imsave(f"{opt.output_dir}/{img_file}", cam)
    cv2.imwrite(f"{opt.output_dir}/{img_file}", fin)

'''
for dataset in ['COD10K']:
    save_path = opt.test_save + dataset + '/'
    os.makedirs(save_path, exist_ok=True)
    # NOTES:
    #  if you plan to inference on your customized dataset without grouth-truth,
    #  you just modify the params (i.e., `image_root=your_test_img_path` and `gt_root=your_test_img_path`)
    #  with the same filepath. We recover the original size according to the shape of grouth-truth, and thus,
    #  the grouth-truth map is unnecessary actually.
    test_loader = test_dataset(image_root='./Dataset/TestDataset/{}/Image/'.format(dataset),
                               gt_root='./Dataset/TestDataset/{}/GT/'.format(dataset),
                               testsize=opt.testsize)
    img_count = 1
    for iteration in range(test_loader.size):
        # load data
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # inference
        _, cam = model(image)
        # reshape and squeeze
        cam = F.upsample(cam, size=gt.shape, mode='bilinear', align_corners=True)
        cam = cam.sigmoid().data.cpu().numpy().squeeze()
        # normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        misc.imsave(save_path+name, cam)
        # evaluate
        mae = eval_mae(numpy2tensor(cam), numpy2tensor(gt))
        # coarse score
        print('[Eval-Test] Dataset: {}, Image: {} ({}/{}), MAE: {}'.format(dataset, name, img_count,
                                                                           test_loader.size, mae))
        img_count += 1

print("\n[Congratulations! Testing Done]")
'''
