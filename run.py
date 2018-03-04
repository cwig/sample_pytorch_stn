import torch
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.nn.modules.module import Module
import matplotlib.pyplot as plt

from torch import nn
import cv2
import os
def convRelu(i, batchNormalization=False, leakyRelu=False):
    nc = 1
    ks = [3, 3, 3, 3, 3, 3, 2]
    ps = [1, 1, 1, 1, 1, 1, 1]
    ss = [1, 1, 1, 1, 1, 1, 1]
    nm = [64, 128, 256, 256, 512, 512, 512]

    cnn = nn.Sequential()

    nIn = nc if i == 0 else nm[i - 1]
    nOut = nm[i]
    cnn.add_module('conv{0}'.format(i),
                   nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
    if batchNormalization:
        cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
    if leakyRelu:
        cnn.add_module('relu{0}'.format(i),
                       nn.LeakyReLU(0.2, inplace=True))
    else:
        cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
    return cnn

def makeCnn():

    cnn = nn.Sequential()
    cnn.add_module('convRelu{0}'.format(0), convRelu(0))
    cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(1), convRelu(1))
    cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(2), convRelu(2, True))
    cnn.add_module('convRelu{0}'.format(3), convRelu(3))
    cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(4), convRelu(4, True))
    cnn.add_module('convRelu{0}'.format(5), convRelu(5))
    cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d(2, 2))
    cnn.add_module('convRelu{0}'.format(6), convRelu(6, True))
    cnn.add_module('pooling{0}'.format(4), nn.MaxPool2d(2, 2))

    return cnn

class PointRegression(Module):
    def __init__(self):
        super(PointRegression, self).__init__()
        self.cnn = makeCnn()
        self.position_linear = nn.Linear(512, 8)
        # position_linear.weight.data.zero_()
        # position_linear.bias.data[0] = 0
        # position_linear.bias.data[1] = 0
        # position_linear.bias.data[2] = 0
        # position_linear.bias.data[3] = 1
        # position_linear.bias.data[4] = 0

    def forward(self, image):
        cnn_out = self.cnn(image)
        cnn_out = torch.squeeze(cnn_out, dim=2)
        cnn_out = torch.squeeze(cnn_out, dim=2)
        out = self.position_linear(cnn_out)
        out = out.view(-1, 2, 4)

        return out

class GridGen(Module):
    def __init__(self, height, width):
        super(GridGen, self).__init__()
        self.height, self.width = height, width
        self.grid = np.zeros( [self.height, self.width, 3], dtype=np.float32)

        self.grid[:,:,0] = np.expand_dims(np.repeat(np.expand_dims(np.arange(self.width)+0.5, 0), repeats = self.height, axis = 0), 0)
        self.grid[:,:,1] = np.expand_dims(np.repeat(np.expand_dims(np.arange(self.height)+0.5, 0), repeats = self.width, axis = 0).T, 0)

        self.grid[:,:,2] = np.ones([self.height, width])
        self.grid = Variable(torch.from_numpy(self.grid.astype(np.float32)), requires_grad=False)

    def forward(self, input):
        out = torch.matmul(input[:,None,None,:,:], self.grid[None,:,:,:,None])
        out = out.squeeze(-1)
        return out

def compute_renorm_matrix(img):
    inv_c = np.array([
        [1.0/img.size(2), 0, 0],
        [0, 1.0/img.size(3), 0],
        [0,0,1]
    ], dtype=np.float32)

    inv_b = np.array([
        [2,0,-1],
        [0,2,-1],
        [0,0, 1]
    ], dtype=np.float32)

    inv_c = Variable(torch.from_numpy(inv_c).type(img.data.type()), requires_grad=False)
    inv_b = Variable(torch.from_numpy(inv_b).type(img.data.type()), requires_grad=False)

    return inv_b.mm(inv_c)

def resample_from_pts(img, pts):

        renorm = compute_renorm_matrix(img)
        pts = torch.cat([
            pts,
            Variable(torch.ones(pts.size(0), 1, 4)).cuda()
        ], dim=1)

        pts = renorm[None,...].matmul(pts)

        output_grid_size = 28
        t = ((np.arange(output_grid_size)+0.5)/output_grid_size)[:,None].astype(np.float32)
        t = np.repeat(t,axis=1, repeats=output_grid_size)
        t = Variable(torch.from_numpy(t), requires_grad=False).cuda()
        s = t.t()

        t = t[:,:,None]
        s = s[:,:,None]

        interpolations = torch.cat([
            (1-t)*(1-s),
            (1-t)*s,
            t*s,
            t*(1-s),
        ], dim=-1)

        grid = interpolations[None,:,:,None,:] * pts[:,None,None,:,:]
        grid = grid.sum(dim=-1)[...,:2]

        resampled = torch.nn.functional.grid_sample(img, grid, mode='bilinear')

        return resampled


class RandomMNISTPosition(object):
    def __init__(self):
        self.s = 16
        self.std = 2
        self.pt_template = torch.Tensor([
            [0,  0, 28, 28],
            [0, 28, 28,  0],
            [1,  1,  1,  1]
        ])

        self.gridgen = GridGen(32, 32)


    def __call__(self, sample):

        img = sample.numpy()[0]
        img = (img - img.min()) / (img.max() - img.min())
        # plt.imshow(img)
        # plt.figure()

        #MNIST digit 28, so translate -14
        translate_to_center = torch.Tensor([
            [1,  0, -14],
            [0,  1, -14],
            [0,  0,  1]
        ])

        #theta = np.random.uniform(-.7, .7)
        theta = np.random.uniform(-np.pi, np.pi)
        rotate = torch.Tensor([
            [ np.cos(theta), np.sin(theta), 0],
            [-np.sin(theta), np.cos(theta), 0],
            [             0,             0, 1]
        ])

        s = np.random.uniform(0.5, 0.7)
        scale = torch.Tensor([
            [s, 0, 0],
            [0, s, 0],
            [0, 0, 1],
        ])

        translate_back = torch.Tensor([
            [1,  0, 14 * s],
            [0,  1, 14 * s],
            [0,  0,      1]
        ])

        tx = np.random.uniform(0, 32 - 28 * s)
        ty = np.random.uniform(0, 32 - 28 * s)
        random_translate = torch.Tensor([
            [1,  0, tx],
            [0,  1, ty],
            [0,  0,  1]
        ])

        final_matrix = random_translate.mm(translate_back).mm(scale).mm(rotate).mm(translate_to_center)

        pts = final_matrix.mm(self.pt_template)

        final_matrix = Variable(final_matrix, requires_grad=False)
        sample = Variable(sample[None,...], requires_grad=False)
        renorm = compute_renorm_matrix(sample)
        inv_final_matrix = torch.inverse(final_matrix)

        grid = self.gridgen(renorm.mm(inv_final_matrix)[None,...])
        grid = grid[...,:2] / grid[...,2:3]

        resampled = torch.nn.functional.grid_sample(sample, grid, mode='bilinear')

        # img = resampled.data.cpu().numpy()[0,0]
        # img = (img - img.min()) / (img.max() - img.min())
        # plt.imshow(img)
        # plt.show()
        return resampled[0].data, pts[:2,:]

if __name__ == "__main__":


    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           RandomMNISTPosition()
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=True, drop_last=True)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('mnist_data', train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           RandomMNISTPosition()
                           # transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=128, shuffle=False, drop_last=True)


    cnn = PointRegression().cuda()
    pt_loss = nn.MSELoss().cuda()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-3)

    lowest_test_loss = float('inf')
    for epoch in xrange(1000):
        cnn.train()
        training_losses = []
        for x in train_loader:
            img = x[0][0]
            target_pts = x[0][1]
            target_label = x[1]



            img = Variable(img, requires_grad=False).cuda()
            target_pts = Variable(target_pts, requires_grad=False).cuda()

            pred_pts = cnn(img)
            loss = pt_loss(pred_pts, target_pts)
            '''
            resampled = resample_from_pts(img, pred_pts)
            target_resampled = resample_from_pts(img, target_pts)
            img = img.data.cpu().numpy()[0,0]
            img = (255*img).astype(np.uint8)
            cv2.imwrite("input.png", img)

            #Not sure why the image is transposed
            img = resampled.data.cpu().numpy()[0,0].T
            img = (255*img).astype(np.uint8)
            cv2.imwrite("output.png", img)
            img = target_resampled.data.cpu().numpy()[0,0].T
            img = (255*img).astype(np.uint8)
            cv2.imwrite("target.png", img)
            '''

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_losses.append(loss.data[0])

        print "Training Loss", np.mean(training_losses)
        cnn.eval()
        test_losses = []
        for x in test_loader:
            img = x[0][0]
            target_pts = x[0][1]
            target_label = x[1]


            img = Variable(img, requires_grad=False, volatile=True).cuda()
            target_pts = Variable(target_pts, requires_grad=False, volatile=True).cuda()

            pred_pts = cnn(img)
            loss = pt_loss(pred_pts, target_pts)
            test_losses.append(loss.data[0])

        test_loss = np.mean(test_losses)
        if lowest_test_loss > test_loss:
            lowest_test_loss = test_loss


            resampled = resample_from_pts(img, pred_pts)
            target_resampled = resample_from_pts(img, target_pts)

            if not os.path.exists("test_samples"):
                os.makedirs("test_samples")

            for i in xrange(img.size(0)):
                im = img.data.cpu().numpy()[i,0]
                im = (255*im).astype(np.uint8)
                cv2.imwrite("test_samples/{}_input.png".format(i), im)

                #Not sure why the image is transposed
                im = resampled.data.cpu().numpy()[i,0].T
                im = (255*im).astype(np.uint8)
                cv2.imwrite("test_samples/{}_output.png".format(i), im)
                im = target_resampled.data.cpu().numpy()[i,0].T
                im = (255*im).astype(np.uint8)
                cv2.imwrite("test_samples/{}_target.png".format(i), im)


        print "Test Loss", test_loss
