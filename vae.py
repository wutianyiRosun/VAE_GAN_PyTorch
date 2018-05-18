import argparse
import torch
from torch.autograd import Variable
import numpy as np
import os
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn

class Normal(object):
    def __init__(self, mu, sigma, log_sigma, v=None, r=None):
        self.mu = mu
        self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
        self.logsigma = log_sigma
        dim = mu.get_shape()
        if v is None:
            v = torch.FloatTensor(*dim)
        if r is None:
            r = torch.FloatTensor(*dim)
        self.v = v
        self.r = r


class Encoder(torch.nn.Module):  #MLP
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class VAE(torch.nn.Module):
    """
    Auto-Encoding Variational Bayes, ICLR2014
    The aim is to minimize the KL divergence between  q_{phi}(z|x^i) and p_{theta}(z|x^i), which is equal to max the varialtional lower
    bound on the marginal likelihood of datapoint i, L(theta, phi; x^i),
    """
    latent_dim = 8

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self._enc_mu = torch.nn.Linear(100, 8)  
        self._enc_log_sigma = torch.nn.Linear(100, 8)

    def _sample_latent(self, h_enc):

        """
        z = g_phi(epsilon, x), epsilon ~ p(epsilon), where epsilon is often sampled from N(0,1)
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)  #mu: 8
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)

        epsilon = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()  #epsilon.shape: 8

        self.z_mean = mu   
        self.z_sigma = sigma
        
        # z = mu +sigma*epsilon ~ N(mu, sigma^2), since epsilon ~ N(0,1)
        return mu + sigma * Variable(epsilon, requires_grad=False).cuda()  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)  #h_enc: 100

        # We let p_{theta}(z|x) be a mulltivariat Gaussian whose distribution paratmeters are computed from epsilon with 
        # a MLP(a fully-connected neural network with a single hidden layer)
        z = self._sample_latent(h_enc)   #1-dim vector: len=8 

        return self.decoder(z)  #reconstruction of input data


def latent_loss(z_mean, z_stddev):
    """
     To maxmize the variational low bound: L(theta, phi; x^i),
     L(theta, phi; x^i) = -D_{KL}( q_{phi}(z|x^i) || p_{theda}(z)) +E_{q_phi(z|x^i)}[ log(p_{theta}(x^i|z) ]
     now, we want to maxmize the varational lower bound L(theta, phi; x^i), which is equal to 
     minimize the D_{KL}( q_{phi}(z|x^i) || p_{theda}(z)), eq.(7) on paper
     and D_{KL}= -0.5 \times {\sum_{j=1}^J (1 + log(sigma_{j}^2) -mu_{j}^2 -sigma_{j}\^2)
     which is equal to minimize the (mu^2 + sigma^2 - log(sigma^2) - 1) 
    """
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="VAE")
    parser.add_argument("--data_dir", default="./data", help="Run on CPU or GPU")
    parser.add_argument("--epoch", type=int, default=100, help="epoch number for training.")
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=str, default="3", help="choose gpu device.")

    return parser.parse_args()

if __name__ == '__main__':
    args = get_arguments()
    print(args)
    if args.cuda:
        print("====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    
    cudnn.enabled = True

    input_dim = 28 * 28  #784
    batch_size = 32

    transform = transforms.Compose(
        [transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST(args.data_dir, train= True,  transform=transform, download= True)

    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    print('Number of samples: ', len(mnist))

    encoder = Encoder(input_dim, 100, 100)  #input: 784, fc1: 100, fc2:100
    decoder = Decoder(8, 100, input_dim)    #input: 8, fc1: 100, fc2: 784
    vae = VAE(encoder, decoder)

    criterion = nn.MSELoss()
    if args.cuda:
        if torch.cuda.device_count()>1:
            print("torch.cuda.device_count()=",torch.cuda.device_count())
            vae = torch.nn.DataParallel(vae).cuda()  #multi-card data parallel
            criterion = criterion.cuda()
        else:
            print("single GPU for training")
            vae = vae.cuda()  #1-card data parallel
            criterion = criterion.cuda()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    for epoch in range(args.epoch):
        for i, data in enumerate(dataloader, 0):
            inputs, classes = data
            inputs, classes = Variable(inputs.resize_(batch_size, input_dim)).cuda(), Variable(classes).cuda()
            optimizer.zero_grad()
            dec = vae(inputs)  #dec is the reconstruction of input data
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll  #the first item is reconstruction error
            loss.backward()
            optimizer.step()
            loss_ = loss.data[0]
        print("====Epoch[{}/{}]: loss: {:.10f}".format(epoch, args.epoch,loss_))

    plt.imshow(vae(inputs).data[0].numpy().reshape(28, 28), cmap='gray')
    plt.show(block=True)
