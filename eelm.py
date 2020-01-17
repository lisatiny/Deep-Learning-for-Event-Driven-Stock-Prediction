from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pickle
import os

class EELM(nn.Module):

    def __init__(self, VOCAB_SIZE, EMBED_SIZE, HID_SIZE, BATCH_SIZE,DEVICE):
        super(EELM, self).__init__()
        self.vocab_size = VOCAB_SIZE
        self.embed_size = EMBED_SIZE
        self.hid_size = HID_SIZE
        self.batch_size = BATCH_SIZE
        self.device = DEVICE

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_size,
            padding_idx=1)

        self.t_1 = nn.Conv2d(in_channels=1, out_channels=self.hid_size, kernel_size=(self.embed_size, self.embed_size))
        self.t_1 = self.t_1.to(self.device)

        self.w_1 = nn.Conv1d(in_channels=1, out_channels=self.hid_size, kernel_size=(2 * self.embed_size))
        self.w_1 = self.w_1.to(self.device)

        self.t_2 = nn.Conv2d(in_channels=1, out_channels=self.hid_size, kernel_size=(self.embed_size, self.embed_size))
        self.t_2 = self.t_2.to(self.device)

        self.w_2 = nn.Conv1d(in_channels=1, out_channels=self.hid_size, kernel_size=(2 * self.embed_size))
        self.w_2 = self.w_2.to(self.device)

        self.t_3 = nn.Conv2d(in_channels=1, out_channels=self.hid_size, kernel_size=(self.hid_size, self.hid_size))
        self.t_3 = self.t_3.to(self.device)

        self.w_3 = nn.Conv1d(in_channels=1, out_channels=self.hid_size, kernel_size=(2 * self.hid_size))
        self.w_3 = self.w_3.to(self.device)

    def forward(self, batch_inputs):
        if self.batch_size != 1 :
            o_1 = self.embedding(batch_inputs[0]).mean(1) # SUBJECT
            p = self.embedding(batch_inputs[1]).mean(1) # RELATION
            o_2 = self.embedding(batch_inputs[2]).mean(1) # OBJECT
            # batch_size * embed_dim
        else :
            o_1 = self.embedding(batch_inputs[0]).mean(0) # SUBJECT
            p = self.embedding(batch_inputs[1]).mean(0) # RELATION
            o_2 = self.embedding(batch_inputs[2]).mean(0) # OBJECT

        o_1 = o_1.unsqueeze(1)
        p = p.unsqueeze(1)
        o_2 = o_2.unsqueeze(1)
        # batch_size * 1 * embed_dim

        op_1 = torch.bmm(o_1.view(self.batch_size,self.embed_size,-1),p.view(self.batch_size,-1,self.embed_size))
        op_2 = torch.bmm(o_2.view(self.batch_size,self.embed_size,-1),p.view(self.batch_size,-1,self.embed_size))
        #[batch_size * embed_size * embed_size]

        r_head_1 = self.t_1(op_1.unsqueeze(1)).squeeze()
        r_tail_1 = self.t_2(op_2.unsqueeze(1)).squeeze()
        # [batch_size * hidden_size]

        cat_op1 = torch.cat((o_1,p),dim=2)
        cat_op2 = torch.cat((o_2,p),dim=2)
        # [batch_size * 1  * 2 x embed_dim]

        r_head_2 = self.w_1(cat_op1).squeeze()
        r_tail_2 = self.w_2(cat_op2).squeeze()
        #[batch_size * hidden_size]

        r_head = torch.tanh(r_head_1 + r_head_2)
        r_tail = torch.tanh(r_tail_1 + r_tail_2)
        # [batch_size * hidden_size]

        r_head = r_head.unsqueeze(1)
        r_tail = r_tail.unsqueeze(1)
        # [batch_size * 1 * hidden_size]

        op_3 = torch.bmm(r_head.view(self.batch_size,self.hid_size,-1),r_tail.view(self.batch_size,-1,self.hid_size))
        # [batch_size * hid_size * hid_size]

        r_top_1 = self.t_3(op_3.unsqueeze(1)).squeeze()
        # [batch_size * hidden_size]

        cat_op3 = torch.cat((r_head,r_tail),dim=2)
        # [batch_size * 1 * 2 x hidden_size]

        r_top_2 = self.w_3(cat_op3).squeeze()
        # [batch_size * hidden_size]

        r_top = torch.tanh(r_top_1 + r_top_2)

        return r_top #[batch_size * hidden_size]

