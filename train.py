import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.functional as tf
import torch.utils.data
import time
from tqdm import tqdm
import model
import argparse
try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
import sys
import os
import pathlib
import scipy.io as io

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. This one shows how to do augmenting during training for a 
    very simple training set    
    """
    def __init__(self):
        """
        Very simple training set made of 200 Gaussians of width between 0.5 and 1.5
        We later augment this with a velocity and amplitude.
        
        Args:
            n_training (int): number of training examples including augmenting
        """
        super(Dataset, self).__init__()

        root = '/scratch1/aasensio/ferran/'
        top = 95

        print("Reading Enhanced_tau_530 - tau")
        tmp = io.readsav(f'{root}Enhanced_tau_530.save')
        self.T_tau = tmp['tempi'].reshape((86, 504*504))
        self.Pe_tau = tmp['epresi'].reshape((86, 504*504))
        self.tau = tmp['tau3'] / 5.0

        print("Reading Enhanced_tau_530 - z")
        tmp = io.readsav(f'{root}Enhanced_530_optiona_rh.save')
        self.T_z = tmp['tg'][0:top, :, :].reshape((top, 504*504))
        self.Pg_z = tmp['pg'][0:top, :, :].reshape((top, 504*504))
        self.z = tmp['z'][0:top] / 1e3

        print("Reading Enhanced_tau_385 - tau")
        tmp = io.readsav(f'{root}Enhanced_tau_385.save')
        self.T_tau = np.concatenate((self.T_tau, tmp['tempi'].reshape((86, 504*504))), axis=1)
        self.Pe_tau = np.concatenate((self.Pe_tau, tmp['epresi'].reshape((86, 504*504))), axis=1)

        print("Reading Enhanced_tau_385 - z")
        tmp = io.readsav(f'{root}Enhanced_optiona_rh.save')
        self.T_z = np.concatenate((self.T_z, tmp['tg'][0:top, :, :].reshape((top, 504*504))), axis=1)
        self.Pg_z = np.concatenate((self.Pg_z, tmp['pg'][0:top, :, :].reshape((top, 504*504))), axis=1)
        
        self.n_training = self.T_z.shape[1]
        
    def __getitem__(self, index):

        T_z = (self.T_z[:, index] - 4000.0) / (15000.0 - 4000.0)
        Pg_z = np.log(self.Pg_z[:, index]) / 10.0

        T_tau = (self.T_tau[:, index] - 4000.0) / (15000.0 - 4000.0)
        Pe_tau = np.log(self.Pe_tau[:, index]) / 10.0

        ind = np.random.randint(low=0, high=86, size=1)[0]
        inp = np.hstack([self.z, T_z, Pg_z, self.tau[ind:ind+1]])
        
        out = np.hstack([T_tau[ind:ind+1], Pe_tau[ind:ind+1]])
        
        return inp.astype('float32'), out.astype('float32')

    def __len__(self):
        return self.n_training

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')
        

class Training(object):
    def __init__(self, batch_size, validation_split=0.2, gpu=0, smooth=0.05):
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu
        self.smooth = smooth
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        self.batch_size = batch_size
        self.validation_split = validation_split        
                
        kwargs = {'num_workers': 2, 'pin_memory': False} if self.cuda else {}        
        
        self.model = model.Network(95*3+1, 100, 2).to(self.device)
        
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model.parameters() if p.requires_grad)))

        self.dataset = Dataset()
        
        # Compute the fraction of data for training/validation
        idx = np.arange(self.dataset.n_training)

        self.train_index = idx[0:int((1-validation_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-validation_split)*self.dataset.n_training):]

        # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
                
        # Data loaders that will inject data during training
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.train_sampler, shuffle=False, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.validation_sampler, shuffle=False, **kwargs)

    def init_optimize(self, epochs, lr, weight_decay, scheduler):

        self.lr = lr
        self.weight_decay = weight_decay        
        print('Learning rate : {0}'.format(lr))
        self.n_epochs = epochs
        
        p = pathlib.Path('trained/')
        p.mkdir(parents=True, exist_ok=True)

        current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
        self.out_name = 'trained/{0}'.format(current_time)

        # Copy model
        file = model.__file__.split('/')[-1]
        shutil.copyfile(model.__file__, '{0}_model.py'.format(self.out_name))
        shutil.copyfile('{0}/{1}'.format(os.path.dirname(os.path.abspath(__file__)), file), '{0}_trainer.py'.format(self.out_name))
        self.file_mode = 'w'

        f = open('{0}_hyper.dat'.format(self.out_name), 'w')
        f.write('Learning_rate       Weight_decay     \n')
        f.write('{0}    {1}'.format(self.lr, self.weight_decay))
        f.close()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.loss_fn = nn.MSELoss().to(self.device)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler, gamma=0.5)

    def optimize(self):
        self.loss = []
        self.loss_val = []
        best_loss = -1e10

        trainF = open('{0}.loss.csv'.format(self.out_name), self.file_mode)

        print('Model : {0}'.format(self.out_name))

        for epoch in range(1, self.n_epochs + 1):            
            self.train(epoch)
            self.test(epoch)
            self.scheduler.step()

            trainF.write('{},{},{}\n'.format(
                epoch, self.loss[-1], self.loss_val[-1]))
            trainF.flush()

            is_best = self.loss_val[-1] < best_loss
            best_loss = min(self.loss_val[-1], best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, filename='{0}.pth'.format(self.out_name))

        trainF.close()

    def train(self, epoch):
        self.model.train()
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        n = 1
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (inputs, outputs) in enumerate(t):
            inputs = inputs.to(self.device)
            outputs = outputs.to(self.device)
            
            self.optimizer.zero_grad()
            out = self.model(inputs)
            
            # Loss
            loss = self.loss_fn(out, outputs)
                    
            loss.backward()

            self.optimizer.step()

            loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            if (NVIDIA_SMI):
                tmp = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                t.set_postfix(loss=loss_avg, lr=current_lr, gpu=tmp.gpu, mem=tmp.memory)
            else:
                t.set_postfix(loss=loss_avg, lr=current_lr)
            
        self.loss.append(loss_avg)

    def test(self, epoch):
        self.model.eval()
        t = tqdm(self.validation_loader)
        n = 1
        loss_avg = 0.0

        with torch.no_grad():
            for batch_idx, (inputs, outputs) in enumerate(t):
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)
                        
                out = self.model(inputs)
            
                # Loss
                loss = self.loss_fn(out, outputs)

                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                
                t.set_postfix(loss=loss_avg)
            
        self.loss_val.append(loss_avg)

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    parser.add_argument('--wd', '--weigth-decay', default=0.0, type=float,
                    metavar='WD', help='Weigth decay')    
    parser.add_argument('--gpu', '--gpu', default=0, type=int,
                    metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float,
                    metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=100, type=int,
                    metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--scheduler', '--scheduler', default=100, type=int,
                    metavar='SCHEDULER', help='Number of epochs before applying scheduler')
    parser.add_argument('--batch', '--batch', default=256, type=int,
                    metavar='BATCH', help='Batch size')
    
    parsed = vars(parser.parse_args())

    deepnet = Training(batch_size=parsed['batch'], gpu=parsed['gpu'], smooth=parsed['smooth'])

    deepnet.init_optimize(parsed['epochs'], lr=parsed['lr'], weight_decay=parsed['wd'], scheduler=parsed['scheduler'])
    deepnet.optimize()