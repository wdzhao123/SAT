import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageEnhance


# several data augumentation strategies
def cv_random_flip(img, label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label


def randomCrop(image, label):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)


def randomRotation(image, label):
    mode = Image.BICUBIC
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
    return image, label


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for i in range(noiseNum):

        randX = random.randint(0, img.shape[0] - 1)

        randY = random.randint(0, img.shape[1] - 1)

        if random.randint(0, 1) == 0:

            img[randX, randY] = 0

        else:

            img[randX, randY] = 255
    return Image.fromarray(img)


class TwoDatasets(data.Dataset):
    def __init__(self,
                 image_root1, gt_root1, image_root2, gt_root2,
                 trainsize
                 ):
        self.trainsize = trainsize
        # get filenames
        self.images1 = [image_root1 + f for f in os.listdir(image_root1) if f.endswith('.jpg')]
        self.gts1 = [gt_root1 + f for f in os.listdir(gt_root1) if f.endswith('.jpg')
                     or f.endswith('.png')]
        self.images2 = [image_root2 + f for f in os.listdir(image_root2) if f.endswith('.jpg')]
        self.gts2 = [gt_root2 + f for f in os.listdir(gt_root2) if f.endswith('.jpg')
                     or f.endswith('.png')]

        # sorted files
        self.images1 = sorted(self.images1)
        self.gts1 = sorted(self.gts1)
        self.images2 = sorted(self.images2)
        self.gts2 = sorted(self.gts2)



        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size1 = len(self.images1)
        self.images2 = self.images2[0:len(self.images1)]
        self.size2 = len(self.images2)

    def __getitem__(self, index):
        image1 = self.rgb_loader(self.images1[index])
        gt1 = self.binary_loader(self.gts1[index])
        name1 = self.images1[index].split('/')[-1]

        image1 = self.img_transform(image1)
        gt1 = self.gt_transform(gt1)

        image2 = self.rgb_loader(self.images2[index])
        gt2 = self.binary_loader(self.gts2[index])
        name2 = self.images2[index].split('/')[-1]

        image2 = self.img_transform(image2)
        gt2 = self.gt_transform(gt2)

        return image1, gt1, image2, gt2, name1, name2

    def filter_files(self):
        assert len(self.images1) == len(self.gts1) and len(self.gts2) == len(self.images2)
        images1 = []
        gts1 = []
        for img_path, gt_path in zip(self.images1, self.gts1):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images1.append(img_path)
                gts1.append(gt_path)
            else:
                print(img_path)
        self.images1 = images1
        self.gts1 = gts1
        images2 = []
        gts2 = []
        for img_path, gt_path in zip(self.images2, self.gts2):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images2.append(img_path)
                gts2.append(gt_path)
        self.images2 = images2
        self.gts2 = gts2

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size1


def two_get_loader(image_root1, gt_root1, image_root2, gt_root2, batchsize, trainsize,
                   shuffle=True, num_workers=12, pin_memory=True):
    dataset = TwoDatasets(image_root1, gt_root1, image_root2, gt_root2, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

