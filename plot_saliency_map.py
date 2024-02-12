import numpy as np
import time

import matplotlib
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

save_dir = '/well/nichols-nvs/users/peo100/cf_net/brnet_toyexample/dcor_net/11022024/'
save_dir = '/home/hutchings/OneDrive/Documents/academic/23-24/phd_work/miniproject2/BR-Net-nature-paper/models/dcor_net/11022024/'

#load the data
train_x = np.load(save_dir + 'train_x.npy')
valid_x = np.load(save_dir + 'valid_x.npy')
train_y = np.load(save_dir + 'train_y.npy')
valid_y = np.load(save_dir + 'valid_y.npy')
train_cf = np.load(save_dir + 'train_cf.npy')
valid_cf = np.load(save_dir + 'valid_cf.npy')

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
            #nn.Sigmoid(), 
            ) 

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


#load model state from file
modelF.load_state_dict(torch.load(save_dir + 'modelF_dcor.pth'))
modelP.load_state_dict(torch.load(save_dir + 'modelP_dcor.pth'))




names = ['train', 'valid']



for name, x in zip(names, [train_x, valid_x]):

  x = torch.tensor(x, dtype=torch.float32).to(device)
  x = x.permute(0,3,1,2)


  # Get the gradients of the output with respect to the input
  x.requires_grad_()

  modelF.eval()
  modelP.eval()
  output = modelP(modelF(x)).sum(dim=0)
  modelF.zero_grad()
  modelP.zero_grad()
  output.backward()

  # Compute the saliency map
  saliency_map = x.grad.abs().squeeze().mean(dim=0).cpu().numpy()

  # Normalize the saliency map
  saliency_map /= saliency_map.max()

  # Plot the original image and the saliency map
  plt.figure(figsize=(10, 5))
  plt.subplot(1, 2, 1)
  plt.imshow(x[0,0,:,:].detach().cpu().numpy())
  plt.title('Original Image')
  plt.axis('off')

  plt.subplot(1, 2, 2)
  plt.imshow(saliency_map)
  plt.title('Saliency Map')
  plt.axis('off')
  # save model
  plt.savefig(save_dir + 'saliency_map_' + name + '.png')
  plt.show()



# dcorr loss
def dcor_loss_calc(X, Y):
  # assume inputs are of size (Batch size, Number of features) ie vectors may need to be .flatten(start_dim=1) first

  # Compute distance matrices
  a = torch.cdist(X.unsqueeze(0), X.unsqueeze(0)).squeeze(0)
  b = torch.cdist(Y.unsqueeze(0), Y.unsqueeze(0)).squeeze(0)

  # Compute double centered distance matrices
  A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean(dim=(0,1), keepdim=True)
  B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean(dim=(0,1), keepdim=True)

  num = (A * B).sum(dim=(0,1)) # = dCovXY * n**2 (n**2 cancels when dividing)
  denom = ((A**2).sum(dim=(0,1))*(B**2).sum(dim=(0,1))).sqrt()# = dVarX * dVarY).sqrt() * n**2 (n**2 cancels when dividing)

  return(num/denom)

modelF.eval()
modelP.eval()

with torch.no_grad():
    train_xx = torch.tensor(train_x, dtype=torch.float32).to(device)
    train_xx = train_xx.permute(0, 3, 1, 2)
    train_yy = torch.tensor(train_y, dtype=torch.float32).to(device)
    train_cf = torch.tensor(train_cf.reshape(-1,1), dtype=torch.float32).to(device)

    F_pred = modelF(train_xx)
    y_pred = modelP(F_pred)

    # Convert predictions to binary values (0 or 1)
    y_pred_binary = torch.round(y_pred.squeeze())
    #print(y_pred.mean(), y_pred.std())

    # Calculate accuracy
    accuracy_trainx = (y_pred_binary == train_yy).sum().item() / train_yy.size(0)

    dcor_train = dcor_loss_calc(F_pred.flatten(start_dim=1), train_cf.flatten(start_dim=1))

    # VALIDATION SET
    valid_xx = torch.tensor(valid_x, dtype=torch.float32).to(device)
    valid_xx = valid_xx.permute(0, 3, 1, 2)
    valid_yy = torch.tensor(valid_y, dtype=torch.float32).to(device)
    valid_cf = torch.tensor(valid_cf.reshape(-1,1), dtype=torch.float32).to(device)

    F_pred = modelF(valid_xx)
    y_pred = modelP(F_pred)
    #print(y_pred.mean(), y_pred.std())

    # Convert predictions to binary values (0 or 1)
    y_pred_binary = torch.round(y_pred.squeeze())

    # Calculate accuracy
    accuracy_validx = (y_pred_binary == valid_yy).sum().item() / valid_yy.size(0)
    dcor_valid = dcor_loss_calc(F_pred.flatten(start_dim=1), valid_cf.flatten(start_dim=1))

print("Accuracy on valid_x: {:.2%}".format(accuracy_validx))
print("Accuracy on train_x: {:.2%}".format(accuracy_trainx))
print("Dcorr on valid_x: {:.4f}".format(dcor_valid))
print("Dcorr on train_x: {:.4f}".format(dcor_train))


# permutation testing
x_tmp = valid_xx
cf_tmp = valid_cf

n = len(x_tmp)
F_pred = modelF(x_tmp)
d_cor_perm_test_list = []
for i in range(10000):
    #permutation of indices 1:n
    F_pred_perm = F_pred[torch.randperm(n),:]
    dist = dcor_loss_calc(cf_tmp.flatten(start_dim=1), F_pred_perm.flatten(start_dim=1))
    d_cor_perm_test_list += [dist.item()]

observation = dcor_loss_calc(F_pred.flatten(start_dim=1), cf_tmp.flatten(start_dim=1)).item()
p_val = (np.array(d_cor_perm_test_list)>observation).mean()
print("p-value: {:.4f}".format(p_val))
if p_val < 0.05:
    print("Reject the null hypothesis that the training set is independent of the confounders")

#plot histogram of d_cor_perm_test_list
plt.hist(d_cor_perm_test_list, bins=1000)
plt.axvline(observation, color='r')
plt.title('Permutation test for dcorr')
plt.xlabel('dcorr')
plt.ylabel('Frequency')
#plt.savefig(save_dir + 'dcorr_perm_test.png')
plt.show()

