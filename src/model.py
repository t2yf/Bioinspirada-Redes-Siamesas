import torch 
import torch.nn as nn
import torchvision.models as models



def instance_model(emb_size = 3):
    """ 
        Instância e altera a última camada para o tamanho de embedding desejado
    """
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    in_features = model.classifier[6].in_features # entrada da última camada (4096)
    model.classifier[6] = nn.Linear(in_features, 128) 

    return model
