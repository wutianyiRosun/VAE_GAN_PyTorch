#######################################################################################################################################
#Ref: 
#   Autoencoding beyond pixels using a learned similarity metric, https://arxiv.org/pdf/1512.09300.pdf
#   Auto-Encoding Variational Bayes, https://arxiv.org/abs/1312.6114
#   Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks https://arxiv.org/pdf/1511.06434.pdf
#######################################################################################################################################
from __future__ import print_function
import argparse
import os
import random
import math
import torch
import torch.nn as nn
import torch.legacy.nn as lnn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import timeit

def get_arguments():
    """
    Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='Dimension of gen filters in first conv_layer')
    parser.add_argument('--ndf', type=int, default=64, help='Dimension of discrim filters in first conv_layer')
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--saveInt', type=int, default=25, help='number of epochs between checkpoints')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='checkpoing', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--cuda', default=True, help="Run on CPU or GPU")
    parser.add_argument('--gpus', type=str, default="3", help="choose gpu device.")
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    return parser.parse_args()

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _Sampler(nn.Module):
    def __init__(self):
        super(_Sampler, self).__init__()
        
    def forward(self,input):
        mu = input[0]
        logvar = input[1]
        
        std = logvar.mul(0.5).exp_() #calculate the STDEV
        if opt.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_() #random normalized noise
        else:
            eps = torch.FloatTensor(std.size()).normal_() #random normalized noise, normal_(mean=0, std=1, *, generator=None)
        eps = Variable(eps)
        return eps.mul(std).add_(mu) # z = mu + std*epsilon ~ N(mu, std)


class _Encoder(nn.Module):
    def __init__(self,imageSize):
        super(_Encoder, self).__init__()
        
        n = math.log2(imageSize)
        
        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        n=int(n)


        self.conv1 = nn.Conv2d(ngf * 2**(n-3), nz, 4)
        self.conv2 = nn.Conv2d(ngf * 2**(n-3), nz, 4)

        self.encoder = nn.Sequential()
        # input is (nc) x 64 x 64
        self.encoder.add_module('input-conv',nn.Conv2d(nc, ngf, 4, 2, 1, bias=False))
        self.encoder.add_module('input-relu',nn.LeakyReLU(0.2, inplace=True))
        for i in range(n-3):  #i= 0, 1, ..., n-2
            # state size. (ngf) x 32 x 32
            self.encoder.add_module('pyramid.{0}-{1}.conv'.format(ngf*2**i, ngf * 2**(i+1)), nn.Conv2d(ngf*2**(i), ngf * 2**(i+1), 4, 2, 1, bias=False))
            self.encoder.add_module('pyramid.{0}.batchnorm'.format(ngf * 2**(i+1)), nn.BatchNorm2d(ngf * 2**(i+1)))
            self.encoder.add_module('pyramid.{0}.relu'.format(ngf * 2**(i+1)), nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf*8) x 4 x 4

    def forward(self,input):
        output = self.encoder(input)
        return [self.conv1(output),self.conv2(output)]


class _netG(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.encoder = _Encoder(imageSize)
        self.sampler = _Sampler()
        
        n = math.log2(imageSize)
        
        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        n=int(n)

        self.decoder = nn.Sequential()  #the G network of DCGAN, input: noise vector Z, output: N x 3 x 64 x 64
        # input is Z, going into a convolution
        self.decoder.add_module('input-conv', nn.ConvTranspose2d(nz, ngf * 2**(n-3), 4, 1, 0, bias=False))
        self.decoder.add_module('input-batchnorm', nn.BatchNorm2d(ngf * 2**(n-3)))
        self.decoder.add_module('input-relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ngf * 2**(n-3)) x 4 x 4

        for i in range(n-3, 0, -1):  #i = n-3, n-2, ..., 1
            self.decoder.add_module('pyramid.{0}-{1}.conv'.format(ngf*2**i, ngf * 2**(i-1)),nn.ConvTranspose2d(ngf * 2**i, 
                                     ngf * 2**(i-1), 4, 2, 1, bias=False))
            self.decoder.add_module('pyramid.{0}.batchnorm'.format(ngf * 2**(i-1)), nn.BatchNorm2d(ngf * 2**(i-1)))
            self.decoder.add_module('pyramid.{0}.relu'.format(ngf * 2**(i-1)), nn.LeakyReLU(0.2, inplace=True))

        self.decoder.add_module('ouput-conv', nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False))
        self.decoder.add_module('output-tanh', nn.Tanh())


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.encoder, input, range(self.ngpu))
            output = nn.parallel.data_parallel(self.sampler, output, range(self.ngpu))
            output = nn.parallel.data_parallel(self.decoder, output, range(self.ngpu))
        else:
            output = self.encoder(input)
            output = self.sampler(output)
            output = self.decoder(output)
        return output
    
    def make_cuda(self):
        self.encoder.cuda()
        self.sampler.cuda()
        self.decoder.cuda()


class _netD(nn.Module):
    def __init__(self, imageSize, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        n = math.log2(imageSize)
        
        assert n==round(n),'imageSize must be a power of 2'
        assert n>=3,'imageSize must be at least 8'
        n=int(n)
        self.main = nn.Sequential()

        # input is (nc) x 64 x 64
        self.main.add_module('input-conv', nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        self.main.add_module('relu', nn.LeakyReLU(0.2, inplace=True))

        # state size. (ndf) x 32 x 32
        for i in range(n-3):
            self.main.add_module('pyramid.{0}-{1}.conv'.format(ngf*2**(i), ngf * 2**(i+1)), nn.Conv2d(ndf * 2 ** (i), ndf * 2 ** (i+1), 4, 2, 1, bias=False))
            self.main.add_module('pyramid.{0}.batchnorm'.format(ngf * 2**(i+1)), nn.BatchNorm2d(ndf * 2 ** (i+1)))
            self.main.add_module('pyramid.{0}.relu'.format(ngf * 2**(i+1)), nn.LeakyReLU(0.2, inplace=True))

        self.main.add_module('output-conv', nn.Conv2d(ndf * 2**(n-3), 1, 4, 1, 0, bias=False))
        self.main.add_module('output-sigmoid', nn.Sigmoid())
        

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)


if __name__ == '__main__':
    opt = get_arguments()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True
    if opt.cuda:
        print("====> Use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
        if torch.cuda.is_available() and not opt.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    
    print("=====> Configuration dataset")
    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                   transform=transforms.Compose([
                                       transforms.Resize(opt.imageSize),
                                       transforms.CenterCrop(opt.imageSize),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])
        )
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 3
 
    print("=====> Building G network of DCGAN (or Decoder of VAE)  ")
    netG = _netG(opt.imageSize,ngpu)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    #print(netG)
    
    
    print("=====> Building D network of DCGAN")
    netD = _netD(opt.imageSize,ngpu)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    #print(netD)

    criterion = nn.BCELoss()
    MSECriterion = nn.MSELoss()

    input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)  #N x 3 x 64 x 64
    noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)                         #N x 100 x 1 x 1
    fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
    label = torch.FloatTensor(opt.batchSize)
    real_label = 1
    fake_label = 0

    if opt.cuda:
        netD.cuda()
        netG.make_cuda()
        criterion.cuda()
        MSECriterion.cuda()
        input, label = input.cuda(), label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    input = Variable(input)
    label = Variable(label)
    noise = Variable(noise)
    fixed_noise = Variable(fixed_noise)

    print("=====> Setup optimizer")
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    gen_win = None
    rec_win = None
    start = timeit.default_timer()
    print("=====> Begin training")
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ###################################################################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###################################################################
            # train with real
            netD.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            input.data.resize_(real_cpu.size()).copy_(real_cpu)
            label.data.resize_(real_cpu.size(0)).fill_(real_label)  #real_label=1
            
            #print("input size of netDs: ", input.size())
            output = netD(input)
            #print("output size of netDs: ", output.size())
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake, taking the noise vector z as the input of D network
            noise.data.resize_(batch_size, nz, 1, 1)
            noise.data.normal_(0, 1)


            print("input size of netG.decorer : ", noise.size())
            gen = netG.decoder(noise)
            print("output size of netG.decorer : ", gen.size())
            label.data.fill_(fake_label)
            output = netD(gen.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()
            ###################################################
            # (2) Update G network which is the decoder of VAE
            ###################################################
            
            print("input size of netG.encoder : ", input.size())
            encoded = netG.encoder(input)  # encoded[0] and encoded[1] are the vector Z
            print("output size of netG.encoder : ", encoded[0].size(), encoded[1].size())
            mu = encoded[0]
            logvar = encoded[1]
            
            #D_{KL}= -0.5 \times {\sum_{j=1}^J (1 + log(sigma_{j}^2) -mu_{j}^2 -sigma_{j}\^2)
            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)

            KLD = torch.sum(KLD_element).mul_(-0.5)
            
            sampled = netG.sampler(encoded)
            rec = netG.decoder(sampled)
            
            MSEerr = MSECriterion(rec,input)
            
            VAEerr = KLD + MSEerr;  #MSEerr is the reconstruction error
            VAEerr.backward()
            optimizerG.step()

            ###############################################
            # (3) Update G network: maximize log(D(G(z)))
            ###############################################
            netG.zero_grad()
            label.data.fill_(real_label)  # fake labels are real for generator cost

            rec = netG(input) # this tensor is freed from mem at this point
            output = netD(rec)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

            print('[%d/%d][%d/%d] Loss_VAE: %.4f Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     VAEerr.data[0], errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))

        if epoch!=0:
            state_G={"epoch": epoch+1, "model": netG.state_dict()}
            torch.save(state_G, '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
            state_D={"epoch": epoch+1, "model": netD.state_dict()}
            torch.save(state_D, '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
    
    end = timeit.default_timer()
    print( float(end-start)/3600, 'h')


