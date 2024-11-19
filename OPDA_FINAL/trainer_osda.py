from __future__ import print_function
import argparse
from utils.utils import *
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
from data_loader.get_loader import get_loader
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support as score

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Openset DA')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--net', type=str, default='resnet152', metavar='B',
                    help='which network alex,vgg,res?')
parser.add_argument('--save', action='store_true', default=False,
                    help='save model or not')
parser.add_argument('--save_path', type=str, default='checkpoint/checkpoint', metavar='B',
                    help='checkpoint path')
parser.add_argument('--source_path', type=str, default='./utils/source_list.txt', metavar='B',
                    help='checkpoint path')
parser.add_argument('--target_path', type=str, default='./utils/target_list.txt', metavar='B',
                    help='checkpoint path')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--unit_size', type=int, default=1000, metavar='N',
                    help='unit size of fully connected layer')
parser.add_argument('--update_lower', action='store_true', default=False,
                    help='update lower layer or not')
parser.add_argument('--no_cuda', action='store_true', default=False,
                    help='disable cuda')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

source_data = "/content/drive/MyDrive/OSDA_Papers_adaptation/OPDA_FINAL/data/Source_train.csv"
target_data = "/content/drive/MyDrive/OSDA_Papers_adaptation/OPDA_FINAL/data/Target_train.csv"
evaluation_data = "/content/drive/MyDrive/OSDA_Papers_adaptation/OPDA_FINAL/data/Target_test.csv"
batch_size = args.batch_size
data_transforms = {
    source_data: transforms.Compose([

        transforms.ToTensor(),

    ]),
    target_data: transforms.Compose([

        transforms.ToTensor(),

    ]),
    evaluation_data: transforms.Compose([

        transforms.ToTensor(),

    ]),
}
use_gpu = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
train_loader, test_loader, unique_labels, len_features = get_loader(source_data, target_data, evaluation_data,
                                       data_transforms, batch_size=args.batch_size)
dataset_train = train_loader.load_data()
dataset_test = test_loader

num_class = len(unique_labels)
class_list = unique_labels
#print(len_features)
G, C = get_model(args.net, len_features = len_features, num_class=num_class, unit_size=args.unit_size)
if args.cuda:
    G.cuda()
    C.cuda()
opt_c, opt_g = get_optimizer_visda(args.lr, G, C,
                                   update_lower=args.update_lower)

print(args.save_path)


def train(num_epoch):
    criterion = nn.CrossEntropyLoss().cuda()
    i = 0
    print('train start!')
    for ep in range(num_epoch):
        G.train()
        C.train()
        for batch_idx, data in enumerate(dataset_train):
            i += 1
            if i % 1000 == 0:
                print('iteration %d', i)
            if args.cuda:
                img_s = data['S']
                label_s = data['S_label']
                img_t = data['T']
                img_s, label_s = Variable(img_s.cuda()), \
                                 Variable(label_s.cuda())
                img_t = Variable(img_t.cuda())
            if len(img_t) < batch_size:
                break
            if len(img_s) < batch_size:
                break
            opt_g.zero_grad()
            opt_c.zero_grad()
            feat = G(img_s)
            out_s = C(feat)
            loss_s = criterion(out_s, label_s)
            loss_s.backward()
            target_funk = Variable(torch.FloatTensor(img_t.size()[0], 2).fill_(0.5).cuda())
            p = 1.0
            C.set_lambda(p)
            feat_t = G(img_t)
            out_t = C(feat_t, reverse=True)
            out_t = F.softmax(out_t)
            prob1 = torch.sum(out_t[:, :num_class - 1], 1).view(-1, 1)
            prob2 = out_t[:, num_class - 1].contiguous().view(-1, 1)

            prob = torch.cat((prob1, prob2), 1)
            loss_t = bce_loss(prob, target_funk)
            loss_t.backward()
            opt_g.step()
            opt_c.step()
            opt_g.zero_grad()
            opt_c.zero_grad()

            if batch_idx % args.log_interval == 0:
                print(f'Train Ep: {ep} [{batch_idx * len(data)}/{96} ({100.0 * batch_idx / 96:.2f}%)]'
      f'\tLoss Source: {loss_s.item():.6f}\tLoss Target: {loss_t.item():.6f}')

            if ep > 0 and batch_idx % 1000 == 0:
                test()
                G.train()
                C.train()

        if args.save:
            if not os.path.exists(args.save_path):
                os.mkdir(args.save_path)
            save_model(G, C, args.save_path + '_' + str(ep))


def test():
    G.eval()
    C.eval()
    correct = 0
    size = 0
    per_class_num = np.zeros((num_class))
    per_class_correct = np.zeros((num_class)).astype(np.float32)
    for batch_idx, data in enumerate(dataset_test):
        if args.cuda:
            img_t, label_t = data[0], data[1]
            img_t, label_t = Variable(img_t.cuda(), volatile=True), \
                             Variable(label_t.cuda(), volatile=True)
        feat = G(img_t)
        out_t = C(feat)
        pred = out_t.data.max(1)[1]
        k = label_t.data.size()[0]
        correct += pred.eq(label_t.data).cpu().sum()
        pred = pred.cpu().numpy()
        for t in range(num_class):
            t_ind = np.where(label_t.data.cpu().numpy() == t)
            correct_ind = np.where(pred[t_ind[0]] == t)
            per_class_correct[t] += float(len(correct_ind[0]))
            per_class_num[t] += float(len(t_ind[0]))
        size += k
    per_class_acc = per_class_correct / per_class_num

    print(
        '\nTest set including unknown classes:  Accuracy: {}/{} ({:.0f}%)  ({:.4f}%)\n'.format(
            correct, size,
            100. * correct / size, float(per_class_acc.mean())))
    for ind, category in enumerate(class_list):
        print('%s:%s' % (category, per_class_acc[ind]))

train(args.epochs + 1)
