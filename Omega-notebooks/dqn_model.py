

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple
import torch
import torch.nn as nn
import numpy as np


def prin():
    print("hiooo")

########################################## Embedding layer+ Attention + separate job picking #####################################################
# # currunt_input_size = (20,72,32)
# input_height = max_job_len = 20

# num_gpus_per_machine = 4
# num_machines_per_rack = 4
# num_racks_per_cluster = 2
# resource_size = num_racks_per_cluster + num_machines_per_rack + num_gpus_per_machine #10

# jobqueue_maxlen = 10
# max_gpu_request = 4

# input_width = resource_size+jobqueue_maxlen*max_gpu_request # 50



# ########################################## Embedding layer+ Attention  #####################################################

# class Attention(nn.Module):
#     def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
#         super(Attention, self).__init__(**kwargs)
        
#         self.supports_masking = True
#         self.bias = bias
#         self.feature_dim = feature_dim
#         self.step_dim = step_dim
#         self.features_dim = 0
#         weight = torch.zeros(feature_dim, 1)
#         nn.init.kaiming_uniform_(weight)
#         self.weight = nn.Parameter(weight)
        
#         if bias:
#             self.b = nn.Parameter(torch.zeros(step_dim))
        
#     def forward(self, x, mask=None):
#         feature_dim = self.feature_dim 
#         step_dim = self.step_dim

#         eij = torch.mm( x.contiguous().view(-1, feature_dim), self.weight)

#             # x.contiguous().view(-1, feature_dim), self.weight).view(-1, step_dim)
        
#         if self.bias:
#             eij = eij + self.b
     
#         eij = torch.tanh(eij)
#         a = torch.exp(eij)
        
#         if mask is not None:
#             a = a * mask

#         a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
#         weighted_input = x * torch.unsqueeze(a, -1)
#         return torch.sum(weighted_input, 1)

# class Dueling_DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Dueling_DQN,self).__init__()
#         # print('input_dim, output_dim:',input_dim, output_dim)
#         drp = 0.4
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.vocab_size=50
#         self.embedding_dim=256

        
#         self.embed = nn.Embedding(20,64)
#         self.falt = nn.Flatten()
#         # print('self.embed',self.embed)

#         # print('self.conv :',self.conv ,'input_dim[2]:',input_dim[2])

#         # self.fc_input_dim = self.feature_size()
#         # self.value_stream = nn.Sequential(
#         #     nn.Linear(self.embedding_dim, 64),
#         #     nn.ReLU(),
#         #     nn.Linear(64, 1)
#         # )

       

#         maxlen = 32#
#         # self.embedding_dropout = nn.Dropout2d(0.1)
#         self.lstm1 = nn.LSTM(92160, 128, bidirectional=True, batch_first=True)
#         self.lstm2 = nn.GRU(128*2, 64, bidirectional=True, batch_first=True)

#         self.attention_layer = Attention(128, maxlen)
        
#         self.linear = nn.Linear(64*2 , 64)
#         self.relu = nn.ReLU()
#         self.out = nn.Linear(64, 10)
#     def feature_size(self):
#         return self.embed(torch.autograd.Variable(torch.zeros([self.input_dim[2], self.input_dim[0], self.input_dim[1]]))).view(1, -1).size(1)
        
#     def forward(self, x):
#         x = torch.tensor(x).to(torch.int64)
#         features = self.embed(x) # torch.Size([1,32,9,35])
#         # features = self.falt(features)
#         # print('after flatt',features.size())#with pooling =1
#         # features = features.view(features.size(0), -1)  (1,10080)
#         # values = self.value_stream(features)
#         # print("before flat values.size()",values.size())
#         # values = self.falt(values)
#         features = features.view(features.size(0), -1)  

#         # print("after flat features.size()",features.size())
#         h_lstm, _ = self.lstm1(features)
#         h_lstm, _ = self.lstm2(h_lstm)
#         h_lstm_atten = self.attention_layer(h_lstm)
#         conc = self.relu(self.linear(h_lstm_atten))
#         out = self.out(conc)
#         qvals =  out - out.mean()

#         return qvals

        
    
########################################## Attention #####################################################

# class Attention(nn.Module):
#     def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
#         super(Attention, self).__init__(**kwargs)
        
#         self.supports_masking = True
#         self.bias = bias
#         self.feature_dim = feature_dim
#         self.step_dim = step_dim
#         self.features_dim = 0
#         weight = torch.zeros(feature_dim, 1)
#         nn.init.kaiming_uniform_(weight)
#         self.weight = nn.Parameter(weight)
        
#         if bias:
#             self.b = nn.Parameter(torch.zeros(step_dim))
        
#     def forward(self, x, mask=None):
#         feature_dim = self.feature_dim 
#         step_dim = self.step_dim

#         eij = torch.mm( x.contiguous().view(-1, feature_dim), self.weight)

#             # x.contiguous().view(-1, feature_dim), self.weight).view(-1, step_dim)
        
#         if self.bias:
#             eij = eij + self.b
     
#         eij = torch.tanh(eij)
#         a = torch.exp(eij)
        
#         if mask is not None:
#             a = a * mask

#         a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)
#         weighted_input = x * torch.unsqueeze(a, -1)
#         return torch.sum(weighted_input, 1)

# class Dueling_DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Dueling_DQN,self).__init__()
#         print('input_dim, output_dim:',input_dim, output_dim)
#         drp = 0.4
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
#         # shape=(1, input_dim[0],input_dim[1])
#         # self.embed = torch.nn.Embedding(input_dim[1], 256)

#         #     nn.Conv2d(input_dim[2], 32, kernel_size=3 ,stride = 2),
#         #     nn.ReLU()
#         # )
        
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_dim[2], 32, kernel_size=3 ,stride = 2),
#             nn.ReLU()
#         )
#         print('self.conv',self.conv)
#         # self.maxpool1 = nn.MaxPool2d()

#         # print('self.conv :',self.conv ,'input_dim[2]:',input_dim[2])

#         self.fc_input_dim = self.feature_size()
#         self.value_stream = nn.Sequential(
#             nn.Linear(self.fc_input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1)
#         )

       

#         maxlen = 32#
#         # self.embedding_dropout = nn.Dropout2d(0.1)
#         self.lstm1 = nn.LSTM(self.fc_input_dim, 128, bidirectional=True, batch_first=True)
#         self.lstm2 = nn.GRU(128*2, 64, bidirectional=True, batch_first=True)

#         self.attention_layer = Attention(128, maxlen)
        
#         self.linear = nn.Linear(64*2 , 64)
#         self.relu = nn.ReLU()
#         self.out = nn.Linear(64, 10)
#     def feature_size(self):
#         return self.conv(torch.autograd.Variable(torch.zeros([self.input_dim[2], self.input_dim[0], self.input_dim[1]]))).view(1, -1).size(1)
#     def forward(self, x):
#         # h_embedding = self.embedding(x)
#         # h_embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
#         features = self.conv(x) # torch.Size([1,32,9,35])
#         # features = self.maxpool1(features)  # stride = 2:features.size() torch.Size([1, 32, 3, 11])  , stride=3:([1, 32, 2, 8])
#         # with  no stide features.size() torch.Size([1, 32, 6, 23]),features.size() torch.Size([1, 32, 6, 23])

 
#         print('features.size()',features.size())#with pooling =1  (1,32,9,35)
#         features = features.view(features.size(0), -1)  
#         print('features.size()',features.size())#with pooling =1  (1,10080)

#         values = self.value_stream(features)


#         h_lstm, _ = self.lstm1(features)
#         h_lstm, _ = self.lstm2(h_lstm)
#         h_lstm_atten = self.attention_layer(h_lstm)
#         conc = self.relu(self.linear(h_lstm_atten))
#         out = self.out(conc)
        
#         qvals = values + (out - out.mean())

#         return qvals

        
#  ###################################### Autoencoder (LSTM stack)####################################################

# class Dueling_DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Dueling_DQN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         print('input_dim,output_dim:',input_dim,output_dim)

#         self.conv = nn.Sequential(
#             nn.Conv2d(input_dim[2], 32, kernel_size=3, stride=2),
#             nn.ReLU()
#         )

#         self.fc_input_dim = self.feature_size()
       
# ################################# Encoder

#         # self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
#         # self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
#         # self.flatt = nn.Flatten() # Image grid to single feature vector
#         self.rnn1 = nn.LSTM(input_size=self.fc_input_dim,hidden_size=128,num_layers=1,batch_first=True)
#         self.rnn2 = nn.LSTM(input_size=128,hidden_size=64,num_layers=1,batch_first=True)
#         self.rnn3 = nn.LSTM(input_size=64,hidden_size=32,num_layers=1,batch_first=True)
# ################################# Decoder

#         self.rnn4 = nn.LSTM(input_size=32,hidden_size=32,num_layers=1,batch_first=True)
#         self.rnn5 = nn.LSTM(input_size=32,hidden_size=64,num_layers=1,batch_first=True)
#         self.rnn6 = nn.LSTM(input_size=64,hidden_size=128,num_layers=1,batch_first=True)
        
#         self.output_layer = nn.Linear(128, 10)

#     def feature_size(self):
#         return self.conv(torch.autograd.Variable(torch.zeros([self.input_dim[2], self.input_dim[0], self.input_dim[1]]))).view(1, -1).size(1)


#     def forward(self, x):
                
#         x = self.conv(x)
#         x = x.view(x.size(0), -1)
        
#         # x = x.reshape((1, self.input_dim[0]*self.input_dim[1], 10))
#         # x = self.flatt(x)
#         x, (_, _) = self.rnn1(x)
#         x, (hidden_n, _) = self.rnn2(x)
#         x, (hidden_n, _) = self.rnn3(x)

#         # a= hidden_n.reshape((10, 64))
        
#         # x = a.repeat(self.input_dim, 10)
#         # x = x.reshape((10, self.input_dim, 64))
#         x, (hidden_n, cell_n) = self.rnn4(x)
#         x, (hidden_n, cell_n) = self.rnn5(x)
#         x, (hidden_n, cell_n) = self.rnn6(x)
#         # x = x.reshape((self.input_dim, 128))
        
        
#         return self.output_layer(x)
        
        

























    
#################################################### Autoencoder (CNN stack) !HAS BUGS!####################################################

# class Dueling_DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Dueling_DQN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
        


#         c_hid = 32
#         latent_dim=512
        
#         self.encoder = nn.Sequential(
#             nn.Conv2d(input_dim[2], c_hid, kernel_size=3, padding=1, stride=2), # 32x32 => 16x16
#             nn.GELU(),
#             nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 16x16 => 8x8
#             nn.GELU(),
#             nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2), # 8x8 => 4x4
#             nn.GELU(),
#             nn.Flatten(), # Image grid to single feature vector
#             nn.Linear(1728, latent_dim)
#         )
#         print('self.encoder :',self.encoder )
#         self.linear = nn.Sequential(
#             nn.Linear(latent_dim, 1728),
#             nn.GELU()
#         )
        
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(2*c_hid, 2*c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 4x4 => 8x8
#             nn.GELU(),
#             nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2), # 8x8 => 16x16
#             nn.GELU(),
#             nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
#             nn.GELU(),
#             nn.ConvTranspose2d(c_hid, output_dim, kernel_size=3, output_padding=1, padding=1, stride=2), # 16x16 => 32x32
#             # nn.Softmax(output_dim) # The input images is scaled between -1 and 1, hence the output has to be bounded as well
#         )


#     def forward(self, state):
#         first = self.encoder(state)
#         # first = first.view(first.size(0), -1)
#         second = self.linear(first)
#         second = second.reshape(second.shape[0], -1, 4, 4)
#         third = self.decoder(second)
#         return third

 #################################################### Autoencoder (CNN+Linear)####################################################

# class Dueling_DQN(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(Dueling_DQN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
        
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(input_dim[2], 32, kernel_size=3, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=1, stride=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=1, stride=1),
#             nn.ReLU()
#         )
#         print('self.conv :',self.conv ,'input_dim[2]:',input_dim[2])

#         self.fc_input_dim = self.feature_size()
        
#         print('self.fc_input_dim  :',self.fc_input_dim )
#         # self.value_stream = nn.Sequential(
#         #     nn.Linear(self.fc_input_dim, 512),
#         #     nn.ReLU(),
#         #     nn.Linear(512, 1)
#         # )

#         # self.advantage_stream = nn.Sequential(
#         #     nn.Linear(self.fc_input_dim, 512),
#         #     nn.ReLU(),
#         #     nn.Linear(512, self.output_dim)
#         # )


#         self.encoder = torch.nn.Sequential(
#             nn.Linear(self.fc_input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512,256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 36),
#             nn.ReLU(),
#             nn.Linear(36, 10)

#         )
          
#         # Building an linear decoder with Linear
#         # layer followed by Relu activation function
#         # The Sigmoid activation function
#         # outputs the value between 0 and 1
#         # 9 ==> 784
#         self.decoder = torch.nn.Sequential(
#             nn.Linear(10, 36),
#             nn.ReLU(),
#             nn.Linear(36,64),
#             nn.ReLU(),
#             nn.Linear(64,128),
#             nn.ReLU(),
#             nn.Linear(128,256),
#             nn.ReLU(),
#             nn.Linear(256, 512),
#             nn.ReLU(),
#             nn.Linear(512,self.output_dim),
#             nn.Sigmoid()
#         )
  


#     def feature_size(self):
#         return self.conv(torch.autograd.Variable(torch.zeros([self.input_dim[2], self.input_dim[0], self.input_dim[1]]))).view(1, -1).size(1)



#     def forward(self, state):
#         print('self.conv :',self.conv ,'input_dim[2]:',input_dim[2])

#         features = self.conv(state)
#         features = features.view(features.size(0), -1)
#         values = self.encoder(features)
#         advantages = self.decoder(values)
#         qvals = advantages - advantages.mean()
#         return qvals































################################################################ CNN+LSTM #####################################################################
# class Dueling_DQN(nn.Module):

#     def __init__(self, input_dim, output_dim):
#         super(Dueling_DQN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim

#         self.conv = nn.Sequential(
#             nn.Conv2d(input_dim[2], 32, kernel_size=3, stride=2),
#             nn.ReLU()
#             # nn.LSTM((20,72),32)
#         )
#         self.fc_input_dim = self.feature_size()
        
#         self.value_stream = nn.LSTM(self.fc_input_dim, 512)

# #         self.value_stream = nn.Sequential(
# #             nn.Linear(self.fc_input_dim, 512),
# #             nn.ReLU(),
# #             nn.Linear(512, 1)
# #         )

#         self.advantage_stream = nn.Sequential(
            
#             nn.Linear(512, self.output_dim)
#         )

#     def forward(self, state):
# #         print('state.size()',state.size())
#         features = self.conv(state)
#         features = features.view(features.size(0), -1)
#         values,_ = self.value_stream(features)
#         advantages = self.advantage_stream(values)
        
# #         print('values.size(),advantages.size,advantages.mean():',values.size(),advantages.size(),advantages.mean())
#         qvals = (advantages - advantages.mean())

#         return qvals

#     def feature_size(self):
#         return self.conv(torch.autograd.Variable(torch.zeros([self.input_dim[2], self.input_dim[0], self.input_dim[1]]))).view(1, -1).size(1)






############################################ CNN #########################################################################

import torch
import torch.nn as nn
import numpy as np

class Dueling_DQN(nn.Module):
 
    def __init__(self, input_dim, output_dim):
        super(Dueling_DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim[2], 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.fc_input_dim = self.feature_size()
        self.value_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

    def feature_size(self):
        return self.conv(torch.autograd.Variable(torch.zeros([self.input_dim[2], self.input_dim[0], self.input_dim[1]]))).view(1, -1).size(1)
