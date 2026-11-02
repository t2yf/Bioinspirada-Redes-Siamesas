# Recall@K - https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020/tree/master, by Sungyeon Kim- Proxy Anchor Loss for Deep Metric Learning, CVPR2020
# Changed by t2yf
import torch
from tqdm import tqdm
import torch.nn.functional as F

# normaliza os vetores de embedding -> esfera unitária
def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

# porcentagem de consultas em que pelo menos uma amostra da mesma classe aparece entre os k vizinhos mais próximos
def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels) # labels verdadeiras
    Y : [nb_samples x k] (k predicted labels/neighbours) #matriz labels dos k vizinhos mais proximos
    """

    s = 0 #contador de acertos
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))

# pega batches do dataloader e retorna emb+labels concatenados 
def predict_batchwise(model, dataloader, device):
    model_is_training = model.training #salva estado atual (train, eval)
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(len(ds[0]))]
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for batch in tqdm(dataloader):
            for i, J in enumerate(batch):
                # i = 0: sz_batch * images
                # i = 1: sz_batch * labels
                # i = 2: sz_batch * indices
                if i == 0:
                    # move images to device of model (approximate device)
                    J = model(J.to(device))

                for j in J:
                    A[i].append(j)
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))]

def evaluate_cos(model, dataloader, device): # TODO colocar no log
    # calculate embeddings with model and get targets
    X, T = predict_batchwise(model, dataloader, device) #pega embeddings
    X = l2_norm(X)

    # get predictions by assigning nearest 8 neighbors with cosine
    K = 32
    Y = []
    
    cos_sim = F.linear(X, X)
    Y = T[cos_sim.topk(1 + K)[1][:,1:].cpu()]
    Y = Y.float().cpu()
    
    recall = []
    for k in [1, 5, 10]:
        r_at_k = calc_recall_at_k(T, Y, k)
        k_n_rec = (k, r_at_k)
        recall.append(k_n_rec)
       # print("R@{} : {:.3f}".format(k, 100 * r_at_k)) #TODO mudar para log.txt

    return recall