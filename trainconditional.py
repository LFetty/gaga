import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch.functional as F
from torch import autograd
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from gaga.gaga import *
from dataset import make_dataset
import matplotlib.pyplot as plt
import wandb
from torch.nn import init

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data)#, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('GroupNorm') != -1:  # GroupNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('InstanceNorm') != -1:  # InstanceNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class gaga_lightning(pl.LightningModule):
    def __init__(self, hparams):
        super(gaga_lightning, self).__init__()
        self.hparams = hparams
        self.hparams.min = torch.Tensor([0.0004, -115, -90, -1, -1, -1])
        self.hparams.max = torch.Tensor([0.08, 115, 90, 1, 1, 1])
        self.hparams.mean = torch.Tensor([0.04, 3.3, 0.47, 0.03, 0.01, -0.9])
        self.hparams.std = torch.Tensor([0.0085, 32.5, 20.5, 0.34, 0.24, 0.12])
        self.generator = Generator_con(self.hparams, cmin=self.hparams.min, cmax=self.hparams.max)
        self.generator_acc = Generator_con(self.hparams, cmin=self.hparams.min, cmax=self.hparams.max)
        self.generator_acc.eval()
        self.discriminator = Discriminator_con(self.hparams)
        init_weights(self.generator, self.hparams.init)
        init_weights(self.discriminator, self.hparams.init)
        self.accumulate(decay=0)
        print(self.generator)
        print(self.discriminator)
        self.loss = torch.nn.BCELoss()

        self.z = torch.randn(4000, int(self.hparams.batchsize), self.hparams.z_dim)

    def forward(self, z, labels):
        return self.generator(z, labels)

    def accumulate(self, decay=0.999):
        par1 = dict(self.generator_acc.named_parameters())
        par2 = dict(self.generator.named_parameters())

        for k in par1.keys():
            par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

    def gan_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def d_logistic_loss(self, real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        return (real_loss + fake_loss).mean()

    def g_nonsaturating_loss(self, fake_pred):
        loss = F.softplus(-fake_pred).mean()

        return loss

    def d_wasserstein_loss(self, real_pred, fake_pred):
        return -real_pred.mean() + fake_pred.mean()

    def g_wasserstein_loss(self, fake_pred):
        return -fake_pred.mean()

    def d_r1_loss(self, real_pred, real_samples):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_samples, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    def compute_gradient_penalty(self, real_data, fake_data):

        alpha = torch.rand(int(self.hparams.batchsize), 1)#, 1, 1)
        alpha = alpha.expand_as(real_data)
        if self.on_gpu:
            alpha = alpha.cuda()

        # interpolated
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated = Variable(interpolated, requires_grad=True)
        if self.on_gpu:
            interpolated = interpolated.cuda()

        prob_interpolated = self.discriminator(interpolated)

        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.on_gpu else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        a = torch.max(gradients_norm - 1, torch.zeros_like(gradients_norm))
        gradient_penalty = (a** 2).mean()

        return gradient_penalty

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def prepare_data(self):

        self.train_dataset = os.listdir(self.hparams.train_dataroot)#make_dataset(self.hparams.train_dataroot, self.hparams.batchsize)
        self.eval_dataset = os.listdir(self.hparams.valid_dataroot)


    def train_dataloader(self):

        array_list= []

        for dataset in self.train_dataset:
            phase, _, _ = phsp.load(os.path.join(self.hparams.train_dataroot, dataset))
            array_list.append(phase)
        array = np.concatenate(array_list, axis=0)
        #labels = np.concatenate(array_list, axis=0)[:, 3]
        self.hparams.mean = array[:, :6].mean(axis=0)
        self.hparams.std = array[:, :6].std(axis=0)
        self.hparams.min = array[:, :6].min(axis=0)
        self.hparams.max = array[:, :6].max(axis=0)

        #array = ((array-self.hparams.min)/(self.hparams.max-self.hparams.min)-0.5)/0.5
        array_data = (array[:, :6]-self.hparams.mean)/self.hparams.std

        array = np.concatenate([array_data, array[:,6:]], axis=1)


        self.data_size = array.shape[0]
        print(array.shape)
        dataloader  = DataLoader(array[:,[0,1,2,3,4,5,6]],
                            batch_size=int(self.hparams.batchsize),
                            num_workers=1,
                            pin_memory=True,
                            # https://discuss.pytorch.org/t/what-is-the-disadvantage-of-using-pin-memory/1702/4
                            shuffle=True,  ## if false ~20% faster, seems identical

                            drop_last=True)

        self.hparams.min = torch.from_numpy(self.hparams.min).cuda()
        self.hparams.max = torch.from_numpy(self.hparams.max).cuda()
        self.hparams.mean = torch.from_numpy(self.hparams.mean).cuda()
        self.hparams.std = torch.from_numpy(self.hparams.std).cuda()
        self.generator.cmin = (self.hparams.min.data - self.hparams.mean.data) / self.hparams.std.data
        self.generator.cmax = (self.hparams.max.data - self.hparams.mean.data) / self.hparams.std.data
        self.generator_acc.cmin = (self.hparams.min.data - self.hparams.mean.data) / self.hparams.std.data
        self.generator_acc.cmax = (self.hparams.max.data - self.hparams.mean.data) / self.hparams.std.data


        #dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=1)#self.hparams.batchsize)

        return dataloader

    def val_dataloader(self):
        dataloader_list = []
        def keys(file):
            return float(file.split('_')[1][:-2])

        self.eval_dataset.sort(key=keys)
        for dataset in self.eval_dataset:
            phase, _, _ = phsp.load(os.path.join(self.hparams.valid_dataroot, dataset))
        #phase, _, _ = phsp.load(os.path.join(self.hparams.train_dataroot,self.train_dataset[0]))
        #print(phase.shape)
            dataloader_list.append(DataLoader(phase[:,[0,1,2,3,4,5,6]], shuffle=False, pin_memory=True, batch_size=int(self.hparams.batchsize)))
        return dataloader_list

    def configure_optimizers(self):
        if self.hparams.optim == 'RMSprop':
            optim_G = torch.optim.RMSprop(self.generator.parameters(), lr=self.hparams.lr)
            optim_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.hparams.lr)

            return [{'optimizer':optim_D, 'frequency': self.hparams.dupdate},
                    {'optimizer':optim_G, 'frequency': self.hparams.gupdate}]

        elif self.hparams.optim == 'AdamW':
            optim_G = torch.optim.AdamW(self.generator.parameters(), lr=self.hparams.lr, betas=(0.0, 0.999),
                                        weight_decay=1e-2)
            optim_D = torch.optim.AdamW(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.0, 0.999),
                                        weight_decay=1e-2)

            schedular_G = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optim_G, max_lr=self.hparams.lr,
                                                                            steps_per_epoch=int(35831121/self.hparams.batchsize),
                                                                            epochs=self.hparams.epochs),
                           'interval': 'step'}

            schedular_D = {'scheduler': torch.optim.lr_scheduler.OneCycleLR(optim_D, max_lr=self.hparams.lr,
                                                                            steps_per_epoch=int(35831121/self.hparams.batchsize),
                                                                            epochs=self.hparams.epochs),
                           'interval': 'step'}


            return [optim_D, optim_G], [schedular_D, schedular_G]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       second_order_closure=None):
        if optimizer_idx == 0:
            optimizer.step()
            optimizer.zero_grad()
        elif optimizer_idx == 1:
            optimizer.step()
            optimizer.zero_grad()
            self.accumulate(decay=0.999)

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_sample = batch[:, :6]
        real_labels = batch[:, 6].long()

        z = torch.randn(real_sample.shape[0], self.hparams.z_dim)
        gen_labels = torch.randint(0, self.hparams.classes, (real_sample.shape[0],))

        if self.on_gpu:
            z = z.cuda(real_sample.device.index)
            gen_labels = gen_labels.cuda(real_sample.device.index)

        if optimizer_idx == 1:

            self.set_requires_grad(self.discriminator, False)

            self.fake_sample = self(z, gen_labels)

            pred = self.discriminator(self.fake_sample, gen_labels)

            if (self.hparams.loss == 'wasserstein'):
                g_loss = self.g_wasserstein_loss(pred)

            elif (self.hparams.loss == 'vanilla'):

                g_loss = self.g_nonsaturating_loss(pred)

            return {'loss':g_loss, 'log':{'g_loss':-pred.mean()}, 'progress_bar':{'g_loss':-pred.mean()}}

        elif optimizer_idx == 0:
            self.set_requires_grad(self.discriminator, True)

            with torch.no_grad():
                self.fake_sample = self(z, gen_labels)

            real_sample.requires_grad = True

            real_pred = self.discriminator(real_sample, real_labels)

            fake_pred = self.discriminator(self.fake_sample.detach(), gen_labels)

            if (self.hparams.loss == 'wasserstein') and (batch_idx%self.hparams.dupdate==0):
                clamp_lower = -0.01#self.params['clamp_lower']
                clamp_upper = 0.01#self.params['clamp_upper']
                for p in self.discriminator.parameters():
                    p.data.clamp_(clamp_lower, clamp_upper)

            if (self.hparams.loss == 'wasserstein'):
                d_loss = self.d_wasserstein_loss(real_pred, fake_pred)

            elif (self.hparams.loss == 'vanilla'):
                d_loss = self.d_logistic_loss(real_pred, fake_pred)

            if (self.hparams.gp == 'r1'):
                gp = self.d_r1_loss(real_pred, real_sample)
            elif (self.hparams.gp == 'mixed'):
                gp = self.compute_gradient_penalty(real_sample, self.fake_sample)
            else:
                gp = 0

            log = {'fake_loss':fake_pred.mean(), 'real_loss':-real_pred.mean()}

            return {'loss':d_loss+gp, 'log':log, 'progress_bar':log}


    def validation_step(self, batch, batch_idx, dataloader_idx):
        real_sample = batch[:, :6]
        real_labels = batch[:, 6].long()


        if self.on_gpu:
            z = self.z[batch_idx].cuda(real_sample.device.index)
        else:
            z = self.z[batch_idx]

        fake_sample = (self.generator_acc(z, real_labels)*self.hparams.std.cuda())+self.hparams.mean.cuda()


        return {'fake_samples': fake_sample, 'real_samples':real_sample}

    def validation_end(self, outputs):

        val_loss = []

        for dataloader_output in outputs:
            fake_samples = torch.reshape(torch.stack([x['fake_samples'] for x in dataloader_output]), (-1,6))
            real_samples = torch.reshape(torch.stack([x['real_samples'] for x in dataloader_output]), (-1,6))


            plt.figure(figsize=(8,8))

            plt.subplot(321)
            loss = 0
            counts_fake, _, _ = plt.hist(fake_samples[:,0].cpu().numpy(), bins=np.arange(self.hparams.min[0].item(),self.hparams.max[0].item(),0.001), alpha=0.5)
            counts_real, _, _ = plt.hist(real_samples[:,0].cpu().numpy(), bins=np.arange(self.hparams.min[0].item(),self.hparams.max[0].item(),0.001), alpha=0.5)

            loss += sum(abs(counts_fake-counts_real))
            plt.subplot(322)
            counts_fake, _, _ =plt.hist(fake_samples[:,1].cpu().numpy(), bins=np.arange(self.hparams.min[1].item(),self.hparams.max[1].item(),1), alpha=0.5)
            counts_real, _, _ =plt.hist(real_samples[:,1].cpu().numpy(), bins=np.arange(self.hparams.min[1].item(),self.hparams.max[1].item(),1), alpha=0.5)
            loss += sum(abs(counts_fake - counts_real))
            plt.subplot(323)
            counts_fake, _, _ =plt.hist(fake_samples[:,2].cpu().numpy(), bins=np.arange(self.hparams.min[2].item(),self.hparams.max[2].item(),1), alpha=0.5)
            counts_real, _, _ =plt.hist(real_samples[:,2].cpu().numpy(), bins=np.arange(self.hparams.min[2].item(),self.hparams.max[2].item(),1), alpha=0.5)
            loss += sum(abs(counts_fake - counts_real))
            plt.subplot(324)
            counts_fake, _, _ =plt.hist(fake_samples[:,3].cpu().numpy(), bins=np.arange(self.hparams.min[3].item(),self.hparams.max[3].item(),0.02), alpha=0.5)
            counts_real, _, _ =plt.hist(real_samples[:,3].cpu().numpy(), bins=np.arange(self.hparams.min[3].item(),self.hparams.max[3].item(),0.02), alpha=0.5)
            loss += sum(abs(counts_fake - counts_real))
            plt.subplot(325)
            counts_fake, _, _ =plt.hist(fake_samples[:,4].cpu().numpy(), bins=np.arange(self.hparams.min[4].item(),self.hparams.max[4].item(),0.02), alpha=0.5)
            counts_real, _, _ =plt.hist(real_samples[:,4].cpu().numpy(), bins=np.arange(self.hparams.min[4].item(),self.hparams.max[4].item(),0.02), alpha=0.5)
            loss += sum(abs(counts_fake - counts_real))
            plt.subplot(326)
            counts_fake, _, _ =plt.hist(fake_samples[:,5].cpu().numpy(), bins=np.arange(self.hparams.min[5].item(),self.hparams.max[5].item(),0.02), alpha=0.5)
            counts_real, _, _ =plt.hist(real_samples[:,5].cpu().numpy(), bins=np.arange(self.hparams.min[5].item(),self.hparams.max[5].item(),0.02), alpha=0.5)
            loss += sum(abs(counts_fake - counts_real))
            val_loss.append(loss)
            self.logger.experiment.log({'plots':wandb.Image(plt)})
            plt.close()

        log = {'MAE_60kV_loss': val_loss[0],'MAE_80kV_loss': val_loss[1],'MAE_100kV_loss': val_loss[2],'MAE_120kV_loss': val_loss[3]}

        return{'val_loss':torch.Tensor([val_loss[0]]), 'log':log, 'progress_bar':log }


def main(args):

    model = gaga_lightning(args)
    name = 'gaga'
    wandblogger = WandbLogger(name=name, project='gaga', entity='lfetty')

    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint('checkpoints', monitor='val_loss', mode='max', period=10)

    trainer = pl.Trainer(max_epochs=args.epochs, gpus=args.ngpus, logger=wandblogger, check_val_every_n_epoch=1,
                         checkpoint_callback=checkpoint_callback)#, distributed_backend='dp')

    trainer.fit(model)

    trainer.save_checkpoint(f'checkpoints/{name}.pt')


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--train_dataroot', type=str, default='data\\train_conditional')
    parser.add_argument('--valid_dataroot', type=str, default='data\\valid_conditional')
    parser.add_argument('--ngpus', type=int, default=1)
    parser.add_argument('--init', type=str, default='kaiming', help='normal|kaiming|xavier|orthogonal')
    parser.add_argument('--optim', type=str, default='RMSprop', help='RMSprop|AdamW')
    parser.add_argument('--act', type=str, default='relu', help='Mish|relu')
    parser.add_argument('--z_dim', type=int, default=6, help='latent vector size')
    parser.add_argument('--x_dim', type=int, default=6, help='output size')
    parser.add_argument('--g_dim', type=int, default=400, help='neurons per layer of generator')
    parser.add_argument('--g_layers', type=int, default=3, help='layers in generator')
    parser.add_argument('--d_dim', type=int, default=400, help='neurons per layer of discriminator')
    parser.add_argument('--d_layers', type=int, default=3, help='layers in discriminator')
    parser.add_argument('--d_norm', type=bool, default=True, help='use norm yes no')
    parser.add_argument('--g_norm', type=bool, default=False, help='use norm yes no')
    parser.add_argument('--g_norm_type', type=str, default='batch', help='layer|batch|instance|group|spectral')
    parser.add_argument('--d_norm_type', type=str, default='spectral', help='layer|batch|instance|group|')
    parser.add_argument('--gupdate', type=int, default=1, help='updates generator each x iterations')
    parser.add_argument('--dupdate', type=int, default=1, help='updates discriminator each x iterations')
    parser.add_argument('--gp', type=str, default='r1', help='r1|mixed')
    parser.add_argument('--loss', type=str, default='vanilla', help='wasserstein|vanilla')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--batchsize', type=int, default=1e4, help='batch size')
    parser.add_argument('--classes', type=int, default=4, help='number of classes')


    args = parser.parse_args()

    main(args)