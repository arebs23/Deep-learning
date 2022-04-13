import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ToyDataset(Dataset):
    def __init__(self, x, y):
        self.x= torch.tensor(x).float().to(device)
        self.y = torch.tensor(y).float().to(device)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)

x = [[1,2],[3,4],[5,6],[7,8]]
y = [[3],[7],[11],[15]]

ds = ToyDataset(x, y)
dl = DataLoader(ds, batch_size=2, shuffle=True)


model = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 1)).to(device)

loss_func = nn.MSELoss()
from torch.optim import SGD
opt = SGD(model.parameters(), lr=0.01)

import time
loss_hist = []
start = time.time()
for _ in range(50):
    for ix, iy in dl:
        opt.zero_grad()
        out = model(ix)
        loss = loss_func(out, iy)
        loss.backward()
        opt.step()
        loss_hist.append(loss.item())

end = time.time()
print(end - start)
val = [[8,9],[10,11],[1.5,2.5]]
model(torch.tensor(val).float().to(device))
print(model.state_dict())
torch.save(model.to('cpu').state_dict(), 'mymodel.pth')