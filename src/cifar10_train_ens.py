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
                      deterministic=True, early_stop_callback=False, logger=logger, checkpoint_callback=False, fast_dev_run=hparams.debug)
    if not hparams.eval:
        trainer.fit(classifier)
    else:
        trainer.test(classifier)
    if hparams.save_model:
        model = classifier.student_models[0] if hparams.num_students else classifier.teacher_model
        th.save(model.state_dict(), 'logs/{}.pt'.format(hparams.classifier))
    
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
    parser.add_argument('--argmax_label', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--eval', type=int, default=0)
    parser.add_argument('--num_eval_students', type=int, default=0)
    parser.add_argument('--num_eval_teachers', type=int, default=0)
    parser.add_argument('--noise_input', type=int, default=0)
    args = parser.parse_args()
    main(args)
