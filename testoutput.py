import os
import time
import json
import random
import argparse
import datetime
import torch.utils
import tqdm
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from data.build import build_transform
from utils.lr_scheduler import build_scheduler
from utils.optimizer import build_optimizer
from utils.logger import create_logger
from utils.utils import  NativeScalerWithGradNormCount, auto_resume_helper, reduce_tensor
from utils.utils import load_checkpoint_ema, load_pretrained_ema, save_checkpoint_ema
from torch.utils.data import Dataset
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count

from timm.utils import ModelEma as ModelEma

from torchvision import datasets, transforms
from PIL import Image
import csv

# dataset
args = {
    'cfg': '/home/ntu/dql/project/classification/cfg.yaml',
    'batch-size': 128,
    'data_path': '/home/ntu/dql/datasets',
    'opts': None,
    'cache-model': 'part',
    'pretrained':'/home/ntu/dql/project/output/training/vssm1_tiny_0230s/20241109103419/ckpt_epoch_299.pth',
    'output': '/home/ntu/dql/project/output/training',
    'tag':time.strftime("%Y%m%d%H%M%S", time.localtime()),
    # 'eval': True,
    # eval参数是student和teacher的区别（因为teacher只需要eval不需要training)
    'model_ema': True,
    'model_ema_decay': 0.9999,
    'model_ema_force_cpu': False,
    'memory_limit_rate': -1
}

class TestDatasetWithoutLabels(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        # 直接获取根目录中所有图像文件路径
        self.samples = sorted([os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = self.samples[index]
        sample = Image.open(path).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, os.path.basename(path)  # 返回图像和文件名
    

def build_dataset_test(config):
    transform = build_transform(True, config)
    dataset_path = os.path.join(config.DATA.DATA_PATH, 'test')
    dataset = TestDatasetWithoutLabels(root=dataset_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size = config.DATA.BATCH_SIZE,shuffle = False)
    return data_loader



def main(config,args):
    # 检查是否需要分布式环境，如果没有则创建一个单GPU的伪进程组
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = '127.0.0.1'  # 本地地址
        os.environ['MASTER_PORT'] = '29500'      # 任意可用端口
        os.environ['RANK'] = '0'                 # 当前进程的 rank 为 0
        os.environ['WORLD_SIZE'] = '1'           # 世界规模为 1
        dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
    # 继续后续代码，如 build_model 的调用

    dataset_test = build_dataset_test(config)
    model = build_model(config)

    torch.cuda.empty_cache()
    model.cuda()
    model_without_ddp = model
    model_ema = None
    optimizer = build_optimizer(config, model, logger)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    loss_scaler = NativeScalerWithGradNormCount()
    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained_ema(config, model_without_ddp, logger, model_ema)
    output_path = 'predictions.csv'
    model.eval()
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Prediction'])  # CSV 文件的表头
        with torch.no_grad():
            for image, filenames in dataset_test:
                images = image.cuda(non_blocking=True)
                outputs = model(images)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()  # 获取预测标签
                for filename, prediction in zip(filenames, predictions):
                    writer.writerow([filename, prediction])


if __name__ == '__main__':
    args = argparse.Namespace(**args)
    config = get_config(args)
    torch.cuda.set_device(0)
    
    cudnn.benchmark = True

    if True: 
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    config.defrost()
    print(config.OUTPUT, flush=True)
    config.freeze()
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")
    
    main(config,args)
