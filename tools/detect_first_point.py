import numpy as np
import torch
import cv2
import os
import os.path as osp
import glob
import argparse
from clrnet.datasets.process import Process
from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes, get_first_point
from clrnet.utils.net_utils import load_network
from pathlib import Path
from tqdm import tqdm


class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Process(cfg.val_process, cfg)
        self.net = build_net(self.cfg)
        self.net = torch.nn.parallel.DataParallel(
            self.net, device_ids=range(1)).cuda()
        self.net.eval()
        load_network(self.net, self.cfg.load_from)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path': img_path, 'ori_img': ori_img})
        return data

    def inference(self, data):
        with torch.no_grad():
            data = self.net(data)
            # print(data.shape)
            # data = self.net.module.get_lanes(data)
            data = self.net.module.heads.get_lanes(data)
            # print(data)
        return data

    def show(self, data):
        out_file = self.cfg.savedir
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        # print(lanes)
        # imshow_lanes(data['ori_img'], lanes, show=self.cfg.show, out_file=out_file)
        first_point_list = get_first_point(lanes)
        return first_point_list

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        # print(data['lanes'])
        if self.cfg.show or self.cfg.savedir:
            first_point_list = self.show(data)
        return first_point_list



def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths


def process(args):
    cfg = Config.fromfile(args.config)
    cfg.show = args.show
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    paths = get_img_paths(args.img)

    total_points_list = []
    for p in tqdm(paths):
        first_point_list = detect.run(p)
        total_points_list.append(first_point_list)
    print(total_points_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img', help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true',
                        help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    process(args)
