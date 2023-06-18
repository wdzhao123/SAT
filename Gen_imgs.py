import torch
import torchvision
from tqdm import tqdm

import lib.networks as vgg
from utils.data_val import *

dataset = 'COD10K'
datatype = 'TrainDataset'


def GenImg(
        data_loader,
        encoder,
        encoder_img,
        decoder_img,
        save_path
):
    for i, (images1, gt1, images2, _, name1, name2) in enumerate(tqdm(data_loader), start=1):
        images1 = images1.cuda()
        images2 = images2.cuda()

        with torch.no_grad():
            lan_fea_sod = encoder(images2)
            lan_fea_cod = encoder(images1)
            lan_fea_cod_img = encoder_img(images1)
            fake_mcps = decoder_img(lan_fea_cod_img, lan_fea_cod, lan_fea_sod)

        os.makedirs(os.path.join(save_path, '{}/Image'.format(dataset)), exist_ok=True)
        os.makedirs(os.path.join(save_path, '{}/GT'.format(dataset)), exist_ok=True)
        os.makedirs(os.path.join(save_path, '{}SOD/Image'.format(dataset)), exist_ok=True)
        torchvision.utils.save_image(
            fake_mcps,
            os.path.join(save_path, '{}/Image'.format(dataset), name1[0]),
            normalize=False
        )
        torchvision.utils.save_image(
            gt1,
            os.path.join(save_path, '{}/GT'.format(dataset), os.path.splitext(name1[0])[0] + '.png'),
            normalize=False
        )
        torchvision.utils.save_image(
            images2,
            os.path.join(save_path, '{}SOD/Image'.format(dataset), name1[0]),
            normalize=False
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1,
                        help='generated images batchsize needs to be set to 1')
    parser.add_argument('--imgsize', type=int, default=352,
                        help='generated images size needs to be set to 352')

    parser.add_argument('--train_cod_img_root', type=str,
                        default='./datasets/COD10K/TrainDataset/Image/',
                        help='COD images root')
    parser.add_argument('--train_cod_gt_root', type=str,
                        default='./datasets/COD10K/TrainDataset/GT/',
                        help='COD gt root')

    parser.add_argument('--train_sod_img_root', type=str,
                        default='./datasets/sorted_DUTS-TR/sorted_DUTS-Image/',
                        help='SOD images root')
    parser.add_argument('--train_sod_gt_root', type=str,
                        default='./datasets/sorted_DUTS-TR/sorted_DUTS-Mask/',
                        help='SOD gt root')

    parser.add_argument('--save_path', type=str,
                        default='./generated_imgs/{}/'
                        .format(datatype),
                        help='the path to save generated images')

    opt = parser.parse_args()

    encoder = vgg.VggEnc().cuda()
    encoder_img = vgg.VggEnc().cuda()
    decoder_img = vgg.VggDecImg().cuda()

    decoder_img.load_state_dict(
        torch.load(
            './snapshot/stage2/decoder_t.pth'
        ),
        strict=True
    )

    encoder_img.load_state_dict(
        torch.load(
            './snapshot/stage2/encoder_t.pth'
        ),
        strict=True
    )

    encoder.load_state_dict(
        torch.load(
            './snapshot/stage2/encoder_p.pth'
        ),
        strict=True
    )

    data_loader = two_get_loader(
        image_root1=opt.train_cod_img_root,
        gt_root1=opt.train_cod_gt_root,
        image_root2=opt.train_sod_img_root,
        gt_root2=opt.train_sod_gt_root,
        batchsize=opt.batchsize,
        trainsize=opt.imgsize,
        num_workers=0,
        shuffle=False
    )

    GenImg(
        data_loader=data_loader,
        encoder=encoder,
        encoder_img=encoder_img,
        decoder_img=decoder_img,
        save_path=opt.save_path,

    )
