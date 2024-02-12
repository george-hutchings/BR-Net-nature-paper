import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

save_dir = '/well/nichols-nvs/users/peo100/cf_net/brnet_toyexample/dcor_net/' + time.strftime("%d%m%Y") + '/'

#make the directory if it is not there
import os
if not os.path.exists(save_dir):
    os.makedirs(save_dir)




lam = 10
epochs = 60000
batch_size = 256




def gkern(kernlen=21, nsig=3):
    import numpy
    import scipy.stats as st

    """Returns a 2D Gaussian kernel array."""

    interval = (2*nsig+1.)/(kernlen)
    x = numpy.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = numpy.diff(st.norm.cdf(x))
    kernel_raw = numpy.sqrt(numpy.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

## Simulate Data as in br example
np.random.seed(1)

N = 640 # number of subjects in a group
labels = np.zeros((N*2,))
labels[N:] = 1

# 2 confounding effects between 2 groups
cf = np.zeros((N*2,))
cf[:N] = np.random.uniform(1,4,size=N)
cf[N:] = np.random.uniform(3,6,size=N)

# 2 major effects between 2 groups
mf = np.zeros((N*2,))
mf[:N] = np.random.uniform(1,4,size=N)
mf[N:] = np.random.uniform(3,6,size=N)

# simulate images
x = np.zeros((N*2,32,32,1))
y = np.zeros((N*2,))
y[N:] = 1
for i in range(N*2):
    x[i,:16,:16,0] = gkern(kernlen=16, nsig=5)*mf[i]
    x[i,16:,:16,0] = gkern(kernlen=16, nsig=5)*cf[i]
    x[i,:16,16:,0] = gkern(kernlen=16, nsig=5)*cf[i]
    x[i,16:,16:,0] = gkern(kernlen=16, nsig=5)*mf[i]
    x[i] = x[i] + np.random.normal(0,0.01,size=(32,32,1))

# shuffle the data
idx = np.random.permutation(len(y))
x = x[idx,:,:,:]
y = y[idx]

class MyDataset(Dataset):
    def __init__(self, data, label, confounders):
        self.data = data
        self.label = label
        self.confounders = confounders
    def __getitem__(self, index):
        return self.data[index], self.label[index], self.confounders[index]
    def __len__(self):
        return len(self.label)
    

train_propotion = 0.8
i = int(N*2*train_propotion)

#convert data to float32
xx = x.astype(np.float32)
yy = y.astype(np.float32)
cfcf = cf.astype(np.float32)

train_x = xx[:i]
valid_x = xx[i:]

train_y = yy[:i]
valid_y = yy[i:]

train_cf = cfcf[:i]
valid_cf = cfcf[i:]

## save the data
np.save(save_dir + 'train_x.npy', train_x)
np.save(save_dir + 'valid_x.npy', valid_x)
np.save(save_dir + 'train_y.npy', train_y)
np.save(save_dir + 'valid_y.npy', valid_y)
np.save(save_dir + 'train_cf.npy', train_cf)
np.save(save_dir + 'valid_cf.npy', valid_cf)

trainset = MyDataset(xx[:i],yy[:i], cfcf[:i])
valset = MyDataset(xx[i:],yy[i:], cfcf[i:])

train_loader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True, pin_memory=True,num_workers=1)
valid_loader = torch.utils.data.DataLoader(valset, batch_size, shuffle=False, pin_memory=True,num_workers=1)

# Define losses

# set torch random seed
torch.manual_seed(1)

lossp_calc = torch.nn.BCELoss()



# Define the models

class modelF1(nn.Module):
    """
    feature predictor
    """
    def __init__(self):
        super(modelF1, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=3, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(2, 4, kernel_size=3, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 8, kernel_size=3, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

    def forward(self, x):
        return self.conv(x)


class modelP1(nn.Module):
    """
    predictor
    """
    def __init__(self):
        super(modelP1, self).__init__()

        self.conv = nn.Sequential(
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1),
            nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)
    
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda" if train_on_gpu else "cpu")


modelF = modelF1()
modelP = modelP1()

modelF = modelF.to(device) # send tensor to device
modelP = modelP.to(device) # send tensor to device

#save model state
modelF_initial_state = modelF.state_dict()
modelP_initial_state = modelP.state_dict()

# dcorr loss
def dcor_loss_calc(X, Y):
  # assume inputs are of size (Batch size, Number of features) ie vectors may need to be .flatten(start_dim=1) first

  # Compute distance matrices
  a = torch.cdist(X.unsqueeze(0), X.unsqueeze(0), compute_mode='donot_use_mm_for_euclid_dist').squeeze(0)
  b = torch.cdist(Y.unsqueeze(0), Y.unsqueeze(0),compute_mode='donot_use_mm_for_euclid_dist').squeeze(0)

  # Compute double centered distance matrices
  A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
  B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()

  num = (A * B).sum() # = dCovXY * n**2 (n**2 cancels when dividing)
  denom = ((A**2).sum()*(B**2).sum()).sqrt() # = (dVarX*n**2 * dVarY*n**2).sqrt()  (n**2 cancels when dividing)

  return(num/denom)

opt = torch.optim.Adam(list(modelF.parameters())+ list(modelP.parameters()), lr=0.0001)


train_loss_dcor_list = []
valid_loss_dcor_list = []
train_lossb_dcor_list = []
valid_lossb_dcor_list = []
train_acc_list = []
valid_acc_list = []

total_time = 0
start_time = time.time()
old_total_loss = np.inf
for epoch in range(epochs):

    train_loss = 0.0
    valid_loss = 0.0
    train_lossb = 0.0
    valid_lossb = 0.0
    train_acc = 0.0
    valid_acc = 0.0


    

    modelF.train()
    modelP.train()
    for x, y, b in train_loader: # b is bias
            #no control group as easier to train
            x,y,b = x.to(device), y.to(device), b.to(device)
            x = x.permute(0,3,1,2)
            y, b = torch.unsqueeze(y,1), torch.unsqueeze(b,1)
            F_pred = modelF(x)
            y_pred = modelP(F_pred)

            #no adversarial loss for the ctrl group
            lossp = lossp_calc(y_pred, y)
            lossb = dcor_loss_calc(F_pred.flatten(start_dim=1), b.flatten(start_dim=1))

            train_loss += lossp.item()
            train_lossb += lossb.item()
            train_acc += (torch.round(y_pred) == y).sum().item() / y.size(0)

            opt.zero_grad(True)
            (lossp + lam*lossb).backward()
            opt.step()




    modelF.eval()
    modelP.eval()
    with torch.no_grad(): #to increase the validation process uses less memory
        for x1, y1, b1 in valid_loader:
            x1,y1,b1 = x1.to(device), y1.to(device), b1.to(device)
            x1 = x1.permute(0,3,1,2)
            y1, b1 = torch.unsqueeze(y1,1), torch.unsqueeze(b1,1)

            F_pred = modelF(x1)
            y_pred = modelP(F_pred)

            lossp = lossp_calc(y_pred, y1)
            lossb = dcor_loss_calc(F_pred.flatten(start_dim=1), b1.flatten(start_dim=1))

            valid_loss += lossp.item()
            valid_lossb += lossb.item()

            valid_loss += lossp.item()
            valid_lossb += lossb.item()
            valid_acc += (torch.round(y_pred) == y1).sum().item() / y1.size(0)

        

    train_loss /= len(train_loader)
    valid_loss /= len(valid_loader)
    train_lossb /= len(train_loader)
    valid_lossb /= len(valid_loader) # hence this is the per batch loss
    train_acc /= len(train_loader)
    valid_acc /= len(valid_loader)


    train_loss_dcor_list += [train_loss]
    valid_loss_dcor_list += [valid_loss]
    train_lossb_dcor_list += [train_lossb]
    valid_lossb_dcor_list += [valid_lossb]
    train_acc_list += [train_acc]
    valid_acc_list += [valid_acc]

    total_loss = valid_loss + lam*valid_lossb
    if total_loss < old_total_loss:
        #save model
        torch.save(modelF.state_dict(), save_dir+'best_modelF_dcor.pth')
        torch.save(modelP.state_dict(), save_dir+'best_modelP_dcor.pth')
        #print model saved
        old_total_loss = total_loss







    if (epoch%100==0):
        elapsed_time = time.time() - start_time
        average_time_per_epoch = elapsed_time / (epoch + 1)
        #print average time per epoch and estimated time remaining
        print('Average time per epoch: {:.0f}m {:.0f}s'.format(average_time_per_epoch // 60, average_time_per_epoch % 60))
        #print estimated time remaining
        print('Estimated remaining time: {:.0f}m {:.0f}s'.format(average_time_per_epoch*(epochs-epoch) // 60, average_time_per_epoch*(epochs-epoch) % 60))


      

        print('Epoch {}/{}'.format(epoch+1, epochs),
                'Train loss: {:.4f}'.format(train_loss),
                'Valid loss: {:.4f}'.format(valid_loss),
                'Train lossb: {:.4f}'.format(train_lossb),
                'Valid lossb: {:.4f}'.format(valid_lossb))

#save model state
torch.save(modelF.state_dict(), save_dir+'modelF_dcor.pth')
torch.save(modelP.state_dict(), save_dir+'modelP_dcor.pth')


# Plotting loss
plt.figure(figsize=(10, 5))
plt.plot(np.array(train_loss_dcor_list), label='Train Loss')
plt.plot(np.array(valid_loss_dcor_list), label='Valid Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.savefig(save_dir + 'loss.png', bbox_inches='tight')
plt.show()

# Plotting lossb
plt.figure(figsize=(10, 5))
plt.plot(np.array(train_lossb_dcor_list), label='Train Lossb')
plt.plot(np.array(valid_lossb_dcor_list), label='Valid Lossb')
plt.xlabel('Epochs')
plt.ylabel('Lossb')
plt.title('Lossb')
plt.legend()
plt.savefig(save_dir + 'lossb.png', bbox_inches='tight')
plt.show()

# Plotting accuracy
plt.figure(figsize=(10, 5))
plt.plot(np.array(train_acc_list), label='Train Accuracy')
plt.plot(np.array(valid_acc_list), label='Valid Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
plt.savefig(save_dir + 'accuracy.png', bbox_inches='tight')
plt.show()



# Plotting total loss
total_loss = [train_loss[i] + lam * train_lossb[i] for i in range(len(train_loss))]
total_val_loss = [valid_loss[i] + lam * valid_lossb[i] for i in range(len(valid_loss))]

plt.figure(figsize=(10, 5))
plt.plot(np.array(total_loss), label='Train Loss + lam * Lossb')
plt.plot(np.array(total_val_loss), label='Valid Loss + lam * Lossb')
plt.xlabel('Epochs')
plt.ylabel('Total Loss')
plt.title('Total Loss')
plt.legend()
plt.savefig(save_dir + 'total_loss.png', bbox_inches='tight')
plt.show()
