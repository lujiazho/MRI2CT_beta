from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict

import os
import json
import cv2
import numpy as np

from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./data/image_pairs.json', 'rt') as f:
            for line in f:
                self.data.extend(json.loads(line))

        self.root = '/ifs/loni/faculty/shi/spectrum/Student_2020/lzhong/MRI2CT'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filepath = item['source']
        target_filepath = item['target']

        source_filepath = os.path.join(self.root, source_filepath)
        target_filepath = os.path.join(self.root, target_filepath)

        source = cv2.imread(source_filepath, cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        target = cv2.imread(target_filepath, cv2.IMREAD_GRAYSCALE)[..., np.newaxis]

        # # Do not forget that OpenCV read images in BGR order.
        # source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        # target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        # # use random images
        # source = np.random.rand(64, 64, 1)
        # target = np.random.rand(64, 64, 1)

        return dict(jpg=target, hint=source)



if __name__ == '__main__':
    # Configs
    # resume_path = './lightning_logs/version_1/checkpoints/epoch=0-step=19999.ckpt'
    batch_size = 4
    logger_freq = 300
    learning_rate = 1e-5

    # First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
    model = create_model('./models/medvdm.yaml').cpu()
    # model.load_state_dict(load_state_dict(resume_path, location='cpu'))
    model.learning_rate = learning_rate

    # Misc
    dataset = MyDataset()
    dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
    logger = ImageLogger(batch_frequency=logger_freq, log_images_kwargs={'sample': True})
    trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger], max_steps=300, accumulate_grad_batches=1)


    # Train!
    trainer.fit(model, dataloader)
