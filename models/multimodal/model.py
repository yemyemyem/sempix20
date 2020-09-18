import torch
import torch.nn as nn
import torchvision.models as models

class MultiModalModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim
        
        # Build resnet
        resnet50 = models.resnet50(pretrained=True)
        modules = list(resnet50.children())[:-1]
        self.resnet50 = nn.Sequential(*modules)

        if bidirectional:
            self.gru = nn.GRU(embedding_dim, int(hidden_dim / 2), bidirectional=True)
        else:
            self.gru = nn.GRU(embedding_dim, hidden_dim)
        
        self.linear = nn.Linear(2048, hidden_dim)

    def forward_cnn(self, img):
        with torch.no_grad():
            result = self.resnet50(img)
        result = result.view(result.shape[0], -1)
        result = self.linear(result)
        return result / torch.norm(result, p=2, dim=1).view(-1,1)

    def forward_cap(self, cap):
        _, hidden = self.gru(cap)
        if self.bidirectional:
            hidden = hidden.unsqueeze(0)
            hidden_fwd = hidden[-1][0]
            hidden_bwd = hidden[-1][1]

            out = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        else:
            out = hidden[-1]

        return out / torch.norm(out, p=2, dim=1).view(-1,1)

    def forward(self, x):
        img, cap = x
        embd_img = self.forward_cnn(img)
        embd_cap = self.forward_cap(cap)
        return torch.cat((embd_img, embd_cap), dim=1)
