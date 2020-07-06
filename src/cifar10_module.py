import torch
import torch as th
import os
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from cifar10_models import *
from pathlib import Path

def get_classifier(classifier, pretrained):
    if classifier == 'vgg11_bn':
        return vgg11_bn(pretrained=pretrained)
    elif classifier == 'vgg13_bn':
        return vgg13_bn(pretrained=pretrained)
    elif classifier == 'vgg16_bn':
        return vgg16_bn(pretrained=pretrained)
    elif classifier == 'vgg19_bn':
        return vgg19_bn(pretrained=pretrained)
    elif classifier == 'resnet18':
        return resnet18(pretrained=pretrained)
    elif classifier == 'resnet34':
        return resnet34(pretrained=pretrained)
    elif classifier == 'resnet50':
        return resnet50(pretrained=pretrained)
    elif classifier == 'densenet121':
        return densenet121(pretrained=pretrained)
    elif classifier == 'densenet161':
        return densenet161(pretrained=pretrained)
    elif classifier == 'densenet169':
        return densenet169(pretrained=pretrained)
    elif classifier == 'mobilenet_v2':
        return mobilenet_v2(pretrained=pretrained)
    elif classifier == 'googlenet':
        return googlenet(pretrained=pretrained)
    elif classifier == 'inception_v3':
        return inception_v3(pretrained=pretrained)
    else:
        raise NameError('Please enter a valid classifier')
        
class CIFAR10_Module(pl.LightningModule):
    def __init__(self, hparams, pretrained=False):
        super().__init__()
        self.hparams = hparams
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]

        self.train_size = len(self.train_dataloader().dataset)
        self.val_size = len(self.val_dataloader().dataset)

        self.criterion = th.nn.CrossEntropyLoss()
        self.teacher_model = get_classifier(hparams.classifier, pretrained=False)
        if not hparams.eval:
            if hparams.num_students:
                # models_path = os.path.expanduser('~/models/cifar10/cifar10_models/state_dicts')
                models_path = os.path.expanduser('~/models/selfens/teachers')
                load_fn =os.path.join(models_path,'{}.pt'.format(hparams.classifier) )
                state_dict = torch.load(load_fn, map_location='cpu')
                self.teacher_model.load_state_dict(state_dict)
                self.student_models = th.nn.ModuleList([
                    get_classifier(hparams.classifier, pretrained=False) for _ in range(hparams.num_students)
                    ])
        else:
            students, teachers = [], []
            addendum = '' if hparams.max_epochs == 100 else '_250'
            for path, lst, num in (
                    (Path('~/models/selfens/students2'+addendum).expanduser(), students, hparams.num_eval_students),
                    (Path('~/models/selfens/teachers2'+addendum).expanduser(), teachers, hparams.num_eval_teachers),
                    ):
                for cmdlog in path.rglob('logs/cmdlog.txt'):
                    params = dict(t[2:].split('=') for t in cmdlog.read_text().split() if t.startswith('--'))
                    if all(str(getattr(hparams, p)) == params[p] for p in ('classifier', 'max_epochs')):
                        lst.append((cmdlog, params))
                lst.sort(key=lambda x: int(x[1]['seed']))
                while len(lst) > num:
                    lst.pop()
                if len(lst) != num:
                    raise ValueError('not enough things to load')
            self._state_dicts = [torch.load(str(p.parent / f'{hparams.classifier}.pt'), map_location='cpu') for p, _ in students + teachers]
            if not self._state_dicts:
                raise ValueError('Must have at least one teacher or student')

    def _loss(self, student_logits, teacher_probs):
        student_log_probs = th.nn.functional.log_softmax(student_logits, -1)
        return -(teacher_probs * student_log_probs).sum(-1)

        
    def forward(self, batch):
        images, labels = batch
        if self.hparams.num_students:
            with th.no_grad():
                self.teacher_model.eval()
                teacher_predictions = self.teacher_model(images).softmax(-1)
            logits = []
            losses = []
            for student in self.student_models:
                student_logits = student(images)
                if self.hparams.argmax_label:
                    student_loss = self.criterion(student_logits, teacher_predictions.argmax(-1))
                else:
                    student_loss = self._loss(student_logits, teacher_predictions).mean()
                logits.append(student_logits)
                losses.append(student_loss)
            loss = sum(losses)
            logits = th.stack(logits, 0)
            if self.hparams.softmax_before_mean:
                logits = logits.softmax(-1)
            logits = logits.mean(0)
            teacher_accuracy = (teacher_predictions.argmax(-1) == labels).float().mean()
        else:
            logits = self.teacher_model(images)
            loss = self.criterion(logits, labels)
            teacher_accuracy = th.zeros_like(loss)

        predictions = logits.argmax(-1)
        accuracy = (predictions == labels).float().mean()
        return loss, accuracy, teacher_accuracy
    
    def training_step(self, batch, batch_nb):
        if self.hparams.noise_input:
            images, labels = batch
            noise = torch.randn_like(images) * images.std() + images.mean()
            batch = noise, labels
        loss, accuracy, teacher_accuracy = self.forward(batch)
        logs = {'loss/train': loss, 'accuracy/train': accuracy, 'teacher_accuracy/train': teacher_accuracy}
        return {'loss': loss, 'log': logs}
        
    def validation_step(self, batch, batch_nb):
        avg_loss, accuracy, teacher_accuracy = self.forward(batch)
        loss = avg_loss * batch[0].size(0)
        corrects = accuracy * batch[0].size(0)
        teacher_corrects = teacher_accuracy * batch[0].size(0)
        logs = {'loss/val': loss, 'corrects': corrects, 'teacher_corrects': teacher_corrects}
        return logs
                
    def validation_epoch_end(self, outputs):
        loss = torch.stack([x['loss/val'] for x in outputs]).sum() / self.val_size
        accuracy = torch.stack([x['corrects'] for x in outputs]).sum() / self.val_size
        teacher_accuracy = torch.stack([x['teacher_corrects'] for x in outputs]).sum() / self.val_size
        logs = {'loss/val': loss, 'accuracy/val': accuracy, 'teacher_accuracy/val': teacher_accuracy}
        if self.hparams.num_students:
            logs['accuracy_ratio/val'] = accuracy / (teacher_accuracy + 1e-6)
        return {'val_loss': loss, 'log': logs}
    
    def test_step(self, batch, batch_nb):
        images, labels = batch
        logits = []
        for sd in self._state_dicts:
            self.teacher_model.load_state_dict(sd)
            logits.append(self.teacher_model(images))
        logits = sum(logits) / len(logits)
        loss = self.criterion(logits, labels) * images.size(0)
        acc = (logits.argmax(-1) == labels).float().mean() * images.size(0)
        logs = {'loss/val': loss, 'accuracy/val': acc}
        return logs
    
    def test_epoch_end(self, outputs):
        loss = torch.stack([x['loss/val'] for x in outputs]).sum() / self.val_size
        accuracy = torch.stack([x['accuracy/val'] for x in outputs]).sum() / self.val_size
        logs = {'loss/val': loss, 'accuracy/val': accuracy}
        return {'log': logs}
        
    def configure_optimizers(self):
        parameters = self.student_models.parameters() if (self.hparams.num_students and not self.hparams.eval) else self.teacher_model.parameters()
        optimizer = torch.optim.SGD(parameters, lr=self.hparams.learning_rate,
                                    weight_decay=self.hparams.weight_decay, momentum=0.9, nesterov=True)
            
        scheduler = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.learning_rate, 
                                                                     steps_per_epoch=self.train_size//self.hparams.batch_size,
                                                                     epochs=self.hparams.max_epochs),
                     'interval': 'step', 'name': 'learning_rate'}
        return [optimizer], [scheduler]
    
    def train_dataloader(self):
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.hparams.data_dir, train=True, transform=transform_train)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader
    
    def val_dataloader(self):
        transform_val = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=self.hparams.data_dir, train=False, transform=transform_val)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, num_workers=4, pin_memory=True)
        return dataloader
    
    def test_dataloader(self):
        return self.val_dataloader()
