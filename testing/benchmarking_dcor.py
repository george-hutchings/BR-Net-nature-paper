import torch
import time
import numpy as np
batch_size = 128
n_repeats = int(5e4)

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda" if train_on_gpu else "cpu")


def dcor_loss_calc(X, Y):
  # assume inputs are of size (Batch size, Number of features) ie vectors may need to be .flatten(start_dim=1) first

  # Compute distance matrices
  a = torch.cdist(X.unsqueeze(0), X.unsqueeze(0)).squeeze(0)
  b = torch.cdist(Y.unsqueeze(0), Y.unsqueeze(0)).squeeze(0)


  # Compute double centered distance matrices
  A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + (a.mean() + 1e-7)
  B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + (b.mean() + 1e-7)

  num = (A * B).sum() # = dCovXY * n**2 (n**2 cancels when dividing)
  denom = ((A**2).sum()*(B**2).sum()).sqrt() # = (dVarX*n**2 * dVarY*n**2).sqrt()  (n**2 cancels when dividing)

  return(num/denom)



#set torch seed
torch.manual_seed(0)
x = torch.randn(128, 1).to(device)
y = torch.randn(128, 32).to(device)
total_times = np.empty(n_repeats)
for i in range(n_repeats):
    start = time.perf_counter()
    tmp = dcor_loss_calc(x, y)
    torch.cuda.synchronize()
    end = time.perf_counter()
    total_times[i] = end - start

print(f"Total time: {total_times.sum():.4f} seconds")
print(f"Average time: {total_times.mean():.4f} seconds")
print(f"Standard deviation: {total_times.std():.4f} seconds")

