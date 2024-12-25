import time
import cv2
import torch
import json
import shutil
import os
import numpy as np
from PIL import Image
from pathlib import Path
import torch.nn.functional as F
from TextBPN.util import canvas as cav
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from TextBPN.dataset.deploy import DeployDataset
from TextBPN.network.textnet import TextNet
from TextBPN.cfglib.config import config as cfg
from TextBPN.cfglib.option import BaseOptions
from TextBPN.util.augmentation import BaseTransform
from TextBPN.util.visualize import visualize_gt
from TextBPN.util.misc import to_device, mkdirs,rescale_result

input_dir = './database/image'
extra_cfg = {
    'net': 'resnet18',
    'scale': 4,
    'exp_name': 'Totaltext',
    'checkepoch': 570,
    'test_size': [640, 960],
    'gpu': '0',
    'dis_threshold': 0.35,
    'cls_threshold': 0.9,
    'viz': True,
    'img_root': input_dir,
    'resume': None,
    'num_workers': 24,
    'cuda': False,
    'mgpu': False,
    'save_dir': './TextBPN/model/',
    'vis_dir': './vis/',
    'log_dir': './logs/',
    'loss': 'CrossEntropyLoss',
    'pretrain': False,
    'verbose': True,
    'max_epoch': 250,
    'lr': 1e-3,
    'lr_adjust': 'fix',
    'stepvalues': [],
    'weight_decay': 0.0,
    'gamma': 0.1,
    'momentum': 0.9,
    'batch_size': 6,
    'optim': 'Adam',
    'save_freq': 5,
    'display_freq': 10,
    'viz_freq': 50,
    'log_freq': 10000,
    'val_freq': 1000,
    'load_memory': False,
    'rescale': 255.0,
    'input_size': 640,
    'start_epoch': 0
}

def update_config(cfg, extra_cfg):
    for k, v in extra_cfg.items():
        cfg[k] = v
        # print(config.gpu)
        cfg.device = torch.device('cuda') if cfg.cuda else torch.device('cpu')

def osmkdir(input_pth):
    if os.path.isdir(input_pth):
        return input_pth
    
    input_dir = '/kaggle/working/input_dir'
    if os.path.exists(input_dir):
        try:
            shutil.rmtree(input_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
            
    os.makedirs(input_dir)
    
    new_path = os.path.join(input_dir, Path(input_pth).name)
    shutil.copy2(input_pth, new_path)  # Use copy2 to preserve metadata
    
    # Return the path of the new folder
    return input_dir

def visualize_detection(image, output_dict, meta=None):
    image_show = image.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])

    cls_preds = F.interpolate(output_dict["fy_preds"], scale_factor=cfg.scale, mode='bilinear')
    cls_preds = cls_preds[0].data.cpu().numpy()

    py_preds = output_dict["py_preds"][1:]
    init_polys = output_dict["py_preds"][0]
    shows = []

    init_py = init_polys.data.cpu().numpy()
    path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name),
                        meta['image_id'][0].split(".")[0] + "_init.png")

    im_show0 = image_show.copy()
    for i, bpts in enumerate(init_py.astype(np.int32)):
        cv2.drawContours(im_show0, [bpts.astype(np.int32)], -1, (255, 255, 0), 2)
        for j, pp in enumerate(bpts):
            if j == 0:
                cv2.circle(im_show0, (int(pp[0]), int(pp[1])), 3, (255, 0, 255), -1)
            elif j == 1:
                cv2.circle(im_show0, (int(pp[0]), int(pp[1])), 3, (0, 255, 255), -1)
            else:
                cv2.circle(im_show0, (int(pp[0]), int(pp[1])), 3, (0, 0, 255), -1)

    cv2.imwrite(path, im_show0)

    for idx, py in enumerate(py_preds):
        im_show = im_show0.copy()
        contours = py.data.cpu().numpy()
        cv2.drawContours(im_show, contours.astype(np.int32), -1, (0, 0, 255), 2)
        for ppts in contours:
            for j, pp in enumerate(ppts):
                if j == 0:
                    cv2.circle(im_show, (int(pp[0]), int(pp[1])), 3, (255, 0, 255), -1)
                elif j == 1:
                    cv2.circle(im_show, (int(pp[0]), int(pp[1])), 3, (0, 255, 255), -1)
                else:
                    cv2.circle(im_show, (int(pp[0]), int(pp[1])), 3, (0, 255, 0), -1)
        path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name),
                             meta['image_id'][0].split(".")[0] + "_{}iter.png".format(idx))
        cv2.imwrite(path, im_show)
        shows.append(im_show)

    show_img = np.concatenate(shows, axis=1)
    show_boundary = cv2.resize(show_img, (320 * len(py_preds), 320))

    cls_pred = cav.heatmap(np.array(cls_preds[0] * 255, dtype=np.uint8))
    dis_pred = cav.heatmap(np.array(cls_preds[1] * 255, dtype=np.uint8))

#     heat_map = np.concatenate([cls_pred*255, dis_pred*255], axis=1)
    heat_map = cls_pred*255
    heat_map = cv2.resize(heat_map, (320, 320))

    return show_boundary, heat_map

def heatmap_to_binary(heatmap, threshold=127):
    heatmap_binary = np.where(heatmap[:,:, 0] >= threshold, 255, 0)
    
    return heatmap_binary

def inference_textBPN(model, test_loader):

    total_time = 0.
    art_results = {}
    for i, (image, meta) in enumerate(test_loader):
        input_dict = dict()
        idx = 0  # test mode can only run with batch_size == 1
        H, W = meta['Height'][idx].item(), meta['Width'][idx].item()
#         print(meta['image_id'], (H, W))

        input_dict['img'] = to_device(image)
        # get detection result
        start = time.time()
        output_dict = model(input_dict)
#         print(output_dict["py_preds"])
        torch.cuda.synchronize()
        end = time.time()
        if i > 0:
            total_time += end - start
            fps = (i + 1) / total_time
        else:
            fps = 0.0
        # visualization
        img_show = image[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)
        
        gt_contour = []
        label_tag = meta['label_tag'][idx].int().cpu().numpy()
        for annot, n_annot in zip(meta['annotation'][idx], meta['n_annotation'][idx]):
            if n_annot.item() > 0:
                gt_contour.append(annot[:n_annot].int().cpu().numpy())

        gt_vis = visualize_gt(img_show, gt_contour, label_tag)
        show_boundary, heat_map = visualize_detection(img_show, output_dict, meta=meta)
#             file_path = os.path.join(cfg.vis_dir, meta['image_id'][idx].split(".")[0]+"_heat_map.json")
        np.set_printoptions(threshold=np.inf)
#             print(heat_map[:,:, 0])
        heatmap_binary = heatmap_to_binary(heat_map)
#         plt.imshow(heatmap_binary)
#         plt.show()
        show_map = np.concatenate([heat_map, gt_vis], axis=1)
        show_map = cv2.resize(show_map, (320 * 3, 320))
        im_vis = np.concatenate([show_map, show_boundary], axis=0)

        contours = output_dict["py_preds"][-1].int().cpu().numpy()
        img_show, contours = rescale_result(img_show, contours, H, W)
#         print('heatmap shape before resize', heatmap_binary.shape)
        
        heatmap_binary = cv2.resize(heatmap_binary.astype(float), (img_show.shape[1], img_show.shape[0]))
        image_text = np.zeros_like(img_show)
    
        for i in range(3):
            image_text[:, :, i] = np.where(heatmap_binary < 127, img_show[:, :, i], 0)
            
        result = {
            'heatmap': heatmap_binary,
            'contours': contours,
            'image_region': image_text
        }
#         print(result.keys())
        art_results[meta['image_id'][0]] = result
        
    return art_results

def focus_text(input_pth,model):
#     if not os.path.isdir(input_pth):
        
    extra_cfg['img_root'] = osmkdir(input_pth)
    update_config(cfg, extra_cfg)
    testset = DeployDataset(
        image_root=cfg.img_root,
        transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds)
    )

    if cfg.cuda:
        cudnn.benchmark = True

    # Data
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    with torch.no_grad():
        res = inference_textBPN(model, test_loader)
        return  Image.fromarray(res[os.path.basename(input_pth)]['image_region'])

update_config(cfg, extra_cfg)
    
ocr_model = TextNet(is_training=False, backbone=cfg.net)
ocr_model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                          'TextBPN_{}_{}.pth'.format(ocr_model.backbone_name, cfg.checkepoch))

ocr_model.load_model(ocr_model_path)
ocr_model = ocr_model.to(cfg.device)  # copy to cuda
ocr_model.eval()        