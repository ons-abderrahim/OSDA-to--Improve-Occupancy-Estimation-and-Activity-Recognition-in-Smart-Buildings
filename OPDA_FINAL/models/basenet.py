from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function




class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        # Save lambda for backward pass
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse the gradient by multiplying with negative lambda
        lambd = ctx.lambd
        return grad_output * -lambd, None

def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)



class Classifier(nn.Module):
    def __init__(self, num_classes=-1):################# You need to change this at each evaluation
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(100, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)
        self.fc3 = nn.Linear(100, num_classes)  # nn.Linear(100, num_classes)

    def set_lambda(self, lambd):
        self.lambd = lambd
    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
            feat = x
            x = self.fc3(x)
        else:
            feat = x
            x = self.fc3(x)
        if return_feat:
            return x, feat
        return x

class ResBase(nn.Module):
    def __init__(self, len_features, option='resnet18', pret=False, unit_size=100):
        super(ResBase, self).__init__()
        
        self.len_features = len_features
        self.conv1 = nn.Conv1d(1, 64, kernel_size=1, bias=False)
        self.conv12 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv22 = nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.conv32 = nn.Conv1d(256, 256, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=1, bias=False)
        self.conv42 = nn.Conv1d(512, 512, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=1, bias=False)
        self.conv52 = nn.Conv1d(1024, 1024, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm1d(1024)
        self.conv6 = nn.Conv1d(1024, 2048, kernel_size=1, bias=False)
        #self.conv62 = nn.Conv1d(2048, 2048, kernel_size=1, bias=False)
        self.bn6 = nn.BatchNorm1d(2048)
        self.dim = self.bn6.num_features*self.len_features 
        self.features = nn.Sequential(self.conv1,self.conv12, self.bn1, self.conv2, self.conv22, self.bn2, self.conv3, \
                                          self.conv32, self.bn3, self.conv4, self.conv42, self.bn4, self.conv5, self.conv52, self.bn5, \
                                          self.conv6, self.bn6)

        # default unit size 100
        self.linear1 = nn.Linear(self.dim, unit_size)
        self.bn7 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear2 = nn.Linear(unit_size, unit_size)
        self.bn8 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear3 = nn.Linear(unit_size, unit_size)
        self.bn9 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear4 = nn.Linear(unit_size, unit_size)
        self.bn10 = nn.BatchNorm1d(unit_size, affine=True)
    def forward(self, x,reverse=False):

        x = self.features(x)
        #print(self.len_features)
        x = x.view(int(x.size(0)), self.dim)
        # best with dropout
        if reverse:
            x = x.detach()


        x = F.dropout(F.relu(self.bn7(self.linear1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn8(self.linear2(x))), training=self.training)
        #x = F.dropout(F.relu(self.bn3(self.linear3(x))), training=self.training)
        #x = F.dropout(F.relu(self.bn4(self.linear4(x))), training=self.training)
        #x = F.relu(self.bn1(self.linear1(x)))
        #x = F.relu(self.bn2(self.linear2(x)))
        return x


class ResClassifier(nn.Module):
    def __init__(self, num_classes=-1, unit_size=1000):
        super(ResClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(unit_size, num_classes)
        )

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x
