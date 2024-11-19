from torchvision import models
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)
'''
class VGGBase(nn.Module):
    # Model VGG
    def __init__(self):
        super(VGGBase, self).__init__()
        model_ft = models.vgg19(pretrained=True)
        print(model_ft)
        mod = list(model_ft.features.children())
        self.lower = nn.Sequential(*mod)
        mod = list(model_ft.classifier.children())
        mod.pop()
        print(mod)
        self.upper = nn.Sequential(*mod)
        self.linear1 = nn.Linear(4096, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.linear2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)

    def forward(self, x, target=False):
        x = self.lower(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        x = self.upper(x)
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))), training=False)
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))), training=False)
        if target:
            return x
        else:
            return x
        return x


class AlexBase(nn.Module):
    def __init__(self):
        super(AlexBase, self).__init__()
        model_ft = models.alexnet(pretrained=True)
        mod = []
        print(model_ft)
        for i in range(18):
            if i < 13:
                mod.append(model_ft.features[i])
        mod_upper = list(model_ft.classifier.children())
        mod_upper.pop()
        # print(mod)
        self.upper = nn.Sequential(*mod_upper)
        self.lower = nn.Sequential(*mod)
        self.linear1 = nn.Linear(4096, 100)
        self.bn1 = nn.BatchNorm1d(100, affine=True)
        self.linear2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100, affine=True)

    def forward(self, x, target=False, feat_return=False):
        x = self.lower(x)
        x = x.view(x.size(0), 9216)
        x = self.upper(x)
        feat = x
        x = F.dropout(F.leaky_relu(self.bn1(self.linear1(x))))
        x = F.dropout(F.leaky_relu(self.bn2(self.linear2(x))))
        if feat_return:
            return feat
        if target:
            return x
        else:
            return x

'''
class Classifier(nn.Module):
    def __init__(self, num_classes=2):
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

class disc(nn.Module):
    def __init__(self,inp=2): #### 
        super(disc, self).__init__()
        self.net = nn.Sequential(nn.Linear(inp, 24),
                                 #nn.BatchNorm1d(48),
                                 nn.ReLU(),
                                 #nn.Dropout(0.2),
                                 nn.Linear(24,12),
                                 #nn.BatchNorm1d(24),
                                 nn.ReLU(),
                                 #nn.Dropout(0.2),
                                 nn.Linear(12,1))
        self.sgm = nn.Sigmoid()

    def forward(self, x):
        x = self.net(x)
        x = self.sgm(x)
        return x

'''
#########################
class ResBase(nn.Module):
    def __init__(self, option='resnet18', pret=False, unit_size=1000):
        super(ResBase, self).__init__()
        self.dim = 2048
        
        model_ft = models.resnet18(pretrained=pret)

        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)
        

           # Define the feature extractor layers
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
        self.bn6 = nn.BatchNorm1d(2048)
        self.feature_layers = nn.Sequential(
            self.conv1, self.conv12, self.bn1,
            self.conv2, self.conv22, self.bn2,
            self.conv3, self.conv32, self.bn3,
            self.conv4, self.conv42, self.bn4,
            self.conv5, self.conv52, self.bn5,
            self.conv6, self.bn6
        )

        # default unit size 100
        self.linear1 = nn.Linear(2048, unit_size)
        self.bn1 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear2 = nn.Linear(unit_size, unit_size)
        self.bn2 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear3 = nn.Linear(unit_size, unit_size)
        self.bn3 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear4 = nn.Linear(unit_size, unit_size)
        self.bn4 = nn.BatchNorm1d(unit_size, affine=True)
    def forward(self, x,reverse=False):

        x = self.features(x)
        x = x.view(x.size(0), self.dim)
        # best with dropout
        if reverse:
            x = x.detach()


        x = F.dropout(F.relu(self.bn1(self.linear1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn2(self.linear2(x))), training=self.training)
        #x = F.dropout(F.relu(self.bn3(self.linear3(x))), training=self.training)
        #x = F.dropout(F.relu(self.bn4(self.linear4(x))), training=self.training)
        #x = F.relu(self.bn1(self.linear1(x)))
        #x = F.relu(self.bn2(self.linear2(x)))
        return x
########################
'''

'''
class ResBase(nn.Module):
    def __init__(self, option='resnet18', pret=False, unit_size=100):
        super(ResBase, self).__init__()

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
        self.in_features = self.bn6.num_features*9 #len_features ########18432/32768
        self.features = nn.Sequential(self.conv1,self.conv12, self.bn1, self.conv2, self.conv22, self.bn2, self.conv3, \
                                              self.conv32, self.bn3, self.conv4, self.conv42, self.bn4, self.conv5, self.conv52, self.bn5, \
                                          self.conv6, self.bn6)


       
    def forward(self, x,reverse=False):

        x = self.features(x)
        x = x.view(x.size(0), 2048)
      

        x = nn.Linear(2048, 100)
      
        return x

    def get_parameters(self):
      parameter_list = list(list(self.conv1.parameters()) + list(self.conv12.parameters()) + list(self.bn1.parameters()) + list(self.conv2.parameters()) + list(self.conv22.parameters()) + list(self.bn2.parameters()) + list(self.conv3.parameters()) + list(self.conv32.parameters()) + list(self.bn3.parameters()) + list(self.conv4.parameters()) + list(self.conv42.parameters()) + list(self.bn4.parameters())+ list(self.conv5.parameters()) + list(self.conv52.parameters()) + list(self.bn5.parameters())++ list(self.conv6.parameters()) + list(self.bn6.parameters()) )

'''
class ResBase(nn.Module):
    def __init__(self, option='resnet18', pret=False, unit_size=1000):
        super(ResBase, self).__init__()
        self.dim = 18432  # This should match the output dimensions after flattening###18432*4

        model_ft = models.resnet18(pretrained=pret)
        mod = list(model_ft.children())
        mod.pop()
        self.features = nn.Sequential(*mod)

        # Define the feature extractor layers
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
        self.bn6 = nn.BatchNorm1d(2048)
        self.feature_layers = nn.Sequential(
            self.conv1, self.conv12, self.bn1,
            self.conv2, self.conv22, self.bn2,
            self.conv3, self.conv32, self.bn3,
            self.conv4, self.conv42, self.bn4,
            self.conv5, self.conv52, self.bn5,
            self.conv6, self.bn6
        )

        # Define the fully connected layers
        self.linear1 = nn.Linear(self.dim, unit_size)
        self.bn_fc1 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear2 = nn.Linear(unit_size, unit_size)
        self.bn_fc2 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear3 = nn.Linear(unit_size, unit_size)
        self.bn_fc3 = nn.BatchNorm1d(unit_size, affine=True)
        self.linear4 = nn.Linear(unit_size, unit_size)
        self.bn_fc4 = nn.BatchNorm1d(unit_size, affine=True)

    def forward(self, x, reverse=False):
        # Apply feature extractor
        x = self.feature_layers(x)
        
        # Calculate the correct dimensions to flatten the tensor
        x = x.view(x.size(0), -1)  # Flatten the tensor

        if reverse:
            x = x.detach()

        # Apply fully connected layers with dropout and batch normalization
        x = F.dropout(F.relu(self.bn_fc1(self.linear1(x))), training=self.training)
        x = F.dropout(F.relu(self.bn_fc2(self.linear2(x))), training=self.training)
        x = F.dropout(F.relu(self.bn_fc3(self.linear3(x))), training=self.training)
        x = F.dropout(F.relu(self.bn_fc4(self.linear4(x))), training=self.training)
        
        return x


class ResClassifier(nn.Module):
    def __init__(self, num_classes=2, unit_size=1000):
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
