import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
import time

# stamp the time
timestamp = time.time()

# check for CUDA
use_cuda = torch.cuda.is_available()

# load in data
raw = pd.read_csv('../dat/schools_w_clusters.csv')
raw = raw[['Cluster ID', 'Id', 'Site name', 'Address', 'Zip', 'Phone']]
raw['Zip'] = raw['Zip'].astype(str)
raw['Phone'] = raw['Phone'].astype(str)

# set up some convienence functions
def extend_to_length(string_to_expand, length):
    extension = '~' * (length-len(string_to_expand))
    return string_to_expand + extension

def record_formatter(record):
    name = extend_to_length(record['Site name'], 95)
    addr = extend_to_length(record['Address'], 43)
    zipp = extend_to_length(record['Zip'], 7)
    phon = extend_to_length(record['Phone'], 9)
    
    strings = list(''.join((name, addr, zipp, phon)))
    characters = np.array(list(map(ord, strings)))
    
    return Variable(torch.from_numpy(characters).float()).view(1,len(characters))

# define the networks
class ae_net(nn.Module):
    def __init__(self, v_size=154, enc_size=50):
        super(ae_net, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(v_size, 400),
            nn.ReLU(True),
            nn.Linear(400, 200),
            nn.ReLU(True),
            nn.Linear(200, 100),
            nn.ReLU(True),
            nn.Linear(100, enc_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(enc_size, 75),
            nn.ReLU(True),
            nn.Linear(75, 100),
            nn.ReLU(True),
            nn.Linear(100, 125),
            nn.ReLU(True),
            nn.Linear(125, v_size)
        )
        
    def autoencode(self, vector):
        return self.decoder(self.encoder(vector))
    
    def encode(self, vector):
        return self.encoder(vector)
        
class disc_net(nn.Module):
    def __init__(self, enc_size=50):
        super(disc_net, self).__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(enc_size*2, 100),
            nn.ReLU(True),
            nn.Linear(100, 50),
            nn.ReLU(True),
            nn.Linear(50,2),
            nn.LogSoftmax(dim=1)
        )

    def discriminate(self, input1, input2):
        output = self.discriminator(torch.cat([input1, input2], dim=1))
        return output

# init learning parameters for autoencoder
learning_rate = 0.001

model1 = ae_net()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)

if use_cuda:
	model1 = model1.cuda()
	criterion = criterion.cuda()

ae_loss = []

model1.train()

# train autoencoder
for epoch in range(1):
    temp_loss = 0
    
    for i in range(raw.shape[0]):
        # build data pairs
        inpt = record_formatter(raw.iloc[i])
        if use_cuda:
                inpt = inpt.cuda()

        # forward
        otpt = model1.autoencode(inpt)
        loss = criterion(otpt, inpt)
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # logging
        temp_loss += loss

    # logging
    ae_loss.append(temp_loss.data[0]/raw.shape[0])
    
model1.eval()

print('\n\nfinished training AE\n\n')

# save autoencoder network parameters
torch.save(model1.state_dict(), ''.join(('../checkpoints/', str(timestamp), '.discriminator_model')))

# init params for training discriminator
learning_rate = 0.001

model2 = disc_net()
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)

if use_cuda:
	model2 = model2.cuda()
	criterion = criterion.cuda()

disc_loss = []
diff = 1
model2.train()

# train discriminator
for epoch in range(1):
    temp_loss = 0
    
    for i in range(raw.shape[0]-diff):
        # build data pairs
        if use_cuda:
                inpt1 = record_formatter(raw.iloc[i])
                inpt1 = inpt1.cuda()
                inpt1 = model1.encode(inpt1)
                inpt2 = record_formatter(raw.iloc[i+diff])
                inpt2 = inpt2.cuda()
                inpt2 = model1.encode(inpt2)
                label = 1 if (raw.iloc[i]['Cluster ID'] == raw.iloc[i+diff]['Cluster ID']) else 0
                label = Variable(torch.LongTensor([label]))
                label = label.cuda()
        else:
                inpt1 = model1.encode(record_formatter(raw.iloc[i]))
                inpt2 = model1.encode(record_formatter(raw.iloc[i+diff]))
                label = 1 if (raw.iloc[i]['Cluster ID'] == raw.iloc[i+diff]['Cluster ID']) else 0
                label = Variable(torch.LongTensor([label]))
        
        # forward
        otpt = model2.discriminate(inpt1, inpt2)
        loss = criterion(otpt, label)
        
        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # logging
        temp_loss += loss

    # logging
    disc_loss.append(temp_loss.data[0]/raw.shape[0])
    
model2.eval()

print('\n\nfinished training discriminator\n\n')

# save autoencoder network parameters
torch.save(model2.state_dict(), ''.join(('../checkpoints/', str(timestamp), '.autoencoder_model')))

# save losses
with open('../checkpoints/losses.dat', 'w') as f:
	f.write('autoencoder losses:\n')
	f.write(str(ae_loss))
	f.write('\ndiscriminator loss:\n')
	f.write(str(disc_loss))
