# Proxy Anchor Loss - https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020/tree/master, by Sungyeon Kim- Proxy Anchor Loss for Deep Metric Learning, CVPR2020

import torch
from torch import nn
import torch.nn.functional as F
import sklearn.preprocessing

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    # Imagens similares label = 1 | Imagens diferentes label = 0
    def forward(self, emb_left, emb_right, label):
        euclidean_dist = F.pairwise_distance(emb_left, emb_right)
        neg = (1-label) * torch.pow(torch.clamp(self.margin - euclidean_dist, min=0.0), 2)
        pos = (label) * torch.pow(euclidean_dist, 2)
        loss = torch.mean(pos+neg) 
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, emb_anchor, emb_positive, emb_negative):
        d_pos = F.pairwise_distance(emb_anchor, emb_positive)
        d_neg = F.pairwise_distance(emb_anchor, emb_negative)
        loss = torch.clamp(d_pos - d_neg + self.margin, min=0.0)
        return loss.mean()


# converte labels do batch em one-hot-encoding
def binarize(T, nb_classes, device):
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).to(device)
    return T

# normaliza os vetores do proxy -> esfera unitária
def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(nn.Module):
    def __init__(self, device,nb_classes, sz_embed, mrg = 0.1, alpha = 32): #TODO nb_classes = 4
        nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = nn.Parameter(torch.randn(nb_classes, sz_embed).to(device)) # cria vetores das classes, nn.Parameter diz que é um parâmetro treinável, atualizado no optimizer.step
        nn.init.kaiming_normal_(self.proxies, mode='fan_out') #inicializa com distribuição normal

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg #margem separação
        self.alpha = alpha #fator escala
        
    def forward(self, X, T, device):
        P = self.proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity entre embeddings do vetor X e proxies P > retorna matriz
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes, device=device) #guarda os pares positivos
        N_one_hot = 1 - P_one_hot # pares negativos
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term     
        
        return loss