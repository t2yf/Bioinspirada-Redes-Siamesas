import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from src.metrics import evaluate_cos
from src.loss import ContrastiveLoss, TripletLoss, Proxy_Anchor
import torch.nn.functional as F

# Contrastive Learning
def train_contrastive(train_loader, model, optimizer, margin, device):
    model.train()
    model.to(device)

    criterion = ContrastiveLoss(margin)
    for l, r, label in train_loader:
        l, r, label = l.to(device), r.to(device), label.to(device)
        optimizer.zero_grad()
        emb_left = model(l)
        emb_right = model(r)
        # normalizar os embeddings
        emb_left =  F.normalize(emb_left, p=2, dim=1)
        emb_right =  F.normalize(emb_right, p=2, dim=1)
        loss = criterion(emb_left, emb_right, label)
        loss.backward()
        optimizer.step()


# Triplet Loss
def train_triplet(train_loader, model, optimizer, margin, device):
    model.train()
    model.to(device)

    criterion = TripletLoss(margin)
    for a, p, n in train_loader:
        a, p, n = a.to(device), p.to(device), n.to(device)
        optimizer.zero_grad()
        emb_anchor = model(a)
        emb_pos = model(p)
        emb_neg = model(n)
        # normalizar os embeddings
        emb_anchor =  F.normalize(emb_anchor, p=2, dim=1)
        emb_pos =  F.normalize(emb_pos, p=2, dim=1)
        emb_neg =  F.normalize(emb_neg, p=2, dim=1)
        loss = criterion(emb_anchor, emb_pos, emb_neg)
        loss.backward()
        optimizer.step()

# Proxy Anchor Loss
def train_proxy_anchor(train_loader, model, optimizer, criterion, device):
    model.train()
    model.to(device)
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        emb = model(x)
        # normalizar o embedding
        emb =  F.normalize(emb, p=2, dim=1)
        loss = criterion(emb, y, device)

        optimizer.zero_grad()
        loss.backward()

        #limita valor do gradiente para [-10,10] 
        #TODO retirar se não for necessário
        torch.nn.utils.clip_grad_value_(criterion.parameters(), 10)

        optimizer.step()


# avaliar com recall@k
def eval(eval_loader, model, device):
    model.eval()
    model.to(device)

    with torch.no_grad():
        recalls = evaluate_cos(model, eval_loader, device)

    return recalls

# TODO imprimir imagens