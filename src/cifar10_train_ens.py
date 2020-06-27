import os
import shutil
import torch
import torch as th
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from cifar10_module import CIFAR10_Module

import cifar10_download

shutil.rmtree('logs', ignore_errors=True)
os.makedirs('logs')

def main(hparams):
    cifar10_download.main()

    if not th.cuda.is_available():
        hparams.cuda = False

    hparams.gpus = '0,' if hparams.cuda else None
    
    seed_everything(hparams.seed)

    # If only train on 1 GPU. Must set_device otherwise PyTorch always store model on GPU 0 first
    if type(hparams.gpus) == str:
        if len(hparams.gpus) == 2: # GPU number and comma e.g. '0,' or '1,'
            torch.cuda.set_device(int(hparams.gpus[0]))
    
    # Model
    classifier = CIFAR10_Module(hparams)
    
    # Trainer
    lr_logger = LearningRateLogger()
    logger = TensorBoardLogger("logs", name=hparams.classifier)
    trainer = Trainer(callbacks=[lr_logger], gpus=hparams.gpus, max_epochs=hparams.max_epochs,
                      deterministic=True, early_stop_callback=False, logger=logger, checkpoint_callback=False)
    trainer.fit(classifier)
    if hparams.save_model:
        th.save(classifier.teacher_model.state_dict(), 'logs/{}.pt'.format(hparams.classifier))
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--classifier', type=str, default='resnet18')
    parser.add_argument('--data_dir', type=str, default=os.path.expanduser('~/data/cifar10/'))
    parser.add_argument('--cuda', default=True) # use None to train on CPU
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--num_students', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--softmax_before_mean', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=0)
    args = parser.parse_args()
    main(args)
