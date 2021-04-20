import numpy as np

import torch
import torch.nn as nn



class GANLoss:

    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samps, fake_samps):
        raise NotImplementedError("dis_loss method has not been implemented")

    def gen_loss(self, real_samps, fake_samps):
        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:
    """ Base class for all conditional losses """

    def __init__(self, dis,gen):
        self.dis = dis
        self.gen = gen

    def loss(self,input,real_samps,num_D,if_l1=True):

        raise NotImplementedError("loss method has not been implemented")


class LSGAN(ConditionalGANLoss):
    def __init__(self,dis,gen):
        from torch.nn import MSELoss
        from torch.nn import L1Loss
        super().__init__(dis,gen)

        self.criterion=MSELoss()
        self.criterion_pix=L1Loss()
        self.criterion_fm=L1Loss()

    def loss(self,input, real_samps, num_D, if_l1=True,l1_lambda=10,fm_lambda=10.):
        loss_D=0
        loss_G=0
        loss_fm=0
        assert real_samps.device == real_samps.device, \
            "Real and Fake samples are not on the same device"
        if type(input)==list:
            dis_input=input[-1]
        else:
            dis_input=input
        fake_samps = self.gen(input)

        device = real_samps.device
        r_preds = self.dis(torch.cat((real_samps, dis_input),dim=1))
        f_preds = self.dis(torch.cat((fake_samps.detach(), dis_input),dim=1))
        for i in range(num_D):
            real_loss = self.criterion(
                r_preds[i][-1],
                torch.ones(r_preds[i][-1].size()).to(device)
            )
            fake_loss = self.criterion(
                f_preds[i][-1],
                torch.zeros(f_preds[i][-1].size()).to(device)
            )
            dis_loss=real_loss+fake_loss
            loss_D+=dis_loss
        # generator
        preds = self.dis(torch.cat((fake_samps, dis_input),dim=1))
        for i in range(num_D):
            for j in range(len(preds[i])):
                loss_fm+=self.criterion_fm(preds[i][j],r_preds[i][j].detach())

            loss_G+=self.criterion(
                preds[i][-1],
                torch.ones(preds[i][-1].size()).to(fake_samps.device)
            )
        loss_G += loss_fm*(1/num_D)*fm_lambda

        if if_l1:
            l1_loss=self.criterion_pix(fake_samps,real_samps)
            loss_G+=l1_lambda*l1_loss

        return loss_D,loss_G, loss_fm




class StandardGAN(ConditionalGANLoss):

    def __init__(self, dis,gen):
        from torch.nn import BCEWithLogitsLoss
        from torch.nn import L1Loss


        super().__init__(dis,gen)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()
        self.criterion_pix = L1Loss()
        self.criterion_fm = L1Loss()

    def loss(self,input,real_samps):
        # small assertion:
        fake_samps=self.gen(input)
        assert real_samps.device == fake_samps.device, \
            "Real and Fake samples are not on the same device"
        if type(input)==list:
            input=input[-1]

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, input)
        f_preds = self.dis(fake_samps.detach(), input)
        real_loss = self.criterion(
            r_preds,
            torch.ones(r_preds.size()).to(device)
        )
        fake_loss = self.criterion(
            f_preds,
            torch.zeros(f_preds.size()).to(device)
        )
        dis_loss = real_loss + fake_loss
        preds = self.dis(fake_samps, input)
        gen_loss = self.criterion(
            preds,
            torch.ones(f_preds.size()).to(fake_samps.device)
        )
        return dis_loss, gen_loss


class HingeGAN(ConditionalGANLoss):

    def __init__(self, dis,gen):
        from torch.nn import L1Loss
        super().__init__(dis,gen)
        self.criterion_l1=L1Loss()

    def loss(self,input,real_samps,if_l1=True,l1_lambda=10.):
        if type(input)==list:
            dis_input=input[-1]
        else:
            dis_input=input
        fake_samps=self.gen(input)
        r_preds = self.dis(real_samps, dis_input)
        f_preds = self.dis(fake_samps.detach(), dis_input)
        dis_loss = (torch.mean(nn.ReLU()(1 - r_preds)) +
                torch.mean(nn.ReLU()(1 + f_preds)))
        preds = self.dis(fake_samps,dis_input)
        gen_loss = -torch.mean(preds)
        if if_l1:
            gen_loss+=l1_lambda*self.criterion_pix(fake_samps,real_samps)
        return dis_loss,gen_loss


class RelativisticAverageHingeGAN(ConditionalGANLoss):

    def __init__(self, dis,gen):
        from torch.nn import L1Loss
        super().__init__(dis,gen)
        self.criterion_pix=L1Loss()

    def loss(self,input,real_samps,if_l1=True,l1_lambda=10.):
        if type(input)==list:
            dis_input=input[-1]
        else:
            dis_input=input
        fake_samps = self.gen(input)
        r_preds = self.dis(real_samps, dis_input)
        f_preds = self.dis(fake_samps.detach(), dis_input)

        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        dis_loss = (torch.mean(nn.ReLU()(1 - r_f_diff))
                + torch.mean(nn.ReLU()(1 + f_r_diff)))

        # Obtain predictions
        # r_preds = self.dis(real_samps, input)
        f_preds = self.dis(fake_samps, dis_input)
        # difference between real and fake:
        r_f_diff = r_preds - torch.mean(f_preds)

        # difference between fake and real samples
        # f_r_diff = f_preds - torch.mean(r_preds)

        # return the loss
        gen_loss = (torch.mean(nn.ReLU()(1 + r_f_diff))
                + torch.mean(nn.ReLU()(1 - f_r_diff)))
        if if_l1:
            gen_loss+=l1_lambda*self.criterion_pix(fake_samps,real_samps)
        return dis_loss,gen_loss


class LogisticGAN(ConditionalGANLoss):
    def __init__(self, dis,gen):
        from torch.nn import L1Loss
        super().__init__(dis,gen)
        self.criterion_pix=L1Loss()
        self.criterion_fm=L1Loss()

    # gradient penalty
    def R1Penalty(self, real_img, input, num_D):

        # TODO: use_loss_scaling, for fp16
        apply_loss_scaling = lambda x: x * torch.exp(x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))
        undo_loss_scaling = lambda x: x * torch.exp(-x * torch.Tensor([np.float32(np.log(2.0))]).to(real_img.device))

        real_img = torch.autograd.Variable(real_img, requires_grad=True)
        real_logit = self.dis(torch.cat((real_img, input),dim=1))
        r1_penalty=0
        for i in range(num_D):
            # real_logit=real_logit[i][-1]
            #real_logit = apply_loss_scaling(torch.sum(real_logit))
            real_grads = torch.autograd.grad(outputs=real_logit[i][-1], inputs=real_img,
                                             grad_outputs=torch.ones(real_logit[i][-1].size()).to(real_img.device),
                                             create_graph=True, retain_graph=True)[0].view(real_img.size(0), -1)
            #real_grads = undo_loss_scaling(real_grads)
            r1_penalty += torch.sum(torch.mul(real_grads, real_grads))
        return r1_penalty

    def loss(self,input,real_samps,num_D,if_l1=True,l1_lambda=10.,r1_gamma=10.0,fm_lambda=10.):
        loss_D=0
        loss_G=0
        loss_fm=0
        if type(input)==list:
            dis_input=input[-1]
        else:
            dis_input=input

        fake_samps=self.gen(input)
        r_preds = self.dis(torch.cat((real_samps, dis_input),dim=1))
        f_preds = self.dis(torch.cat((fake_samps.detach(), dis_input),dim=1))
        for i in range(num_D):
            dis_loss = torch.mean(nn.Softplus()(f_preds[i][-1])) + torch.mean(nn.Softplus()(-r_preds[i][-1]))
            loss_D+=dis_loss
        if r1_gamma != 0.0:
            r1_penalty = self.R1Penalty(real_samps.detach(),dis_input,num_D) * (r1_gamma * 0.5)
            loss_D += r1_penalty

        preds = self.dis(torch.cat((fake_samps, dis_input),dim=1))
        for i in range(num_D):
            for j in range(len(preds)):
                loss_fm+=self.criterion_fm(preds[i][j],r_preds[i][j].detach())

            loss_G+=torch.mean(nn.Softplus()(-preds[i][-1]))
        loss_G+=loss_fm*(1/num_D)*fm_lambda
        if if_l1:
            loss_G+=l1_lambda*self.criterion_pix(fake_samps,real_samps)
        return loss_D,loss_G,loss_fm

class WassersteinLoss(ConditionalGANLoss):
    def __init__(self,dis,gen):
        from torch.nn import L1Loss
        super(WassersteinLoss, self).__init__(dis,gen)
        self.criterion_pix=L1Loss()
        self.criterion_fm=L1Loss()

    def gradient_penalty(self, y,real_img):
        weight = torch.ones(y.size()).to(real_img.device)
        dydx = torch.autograd.grad(outputs=y,
                                   inputs=real_img,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        # 将dy/dx转换为batch，n
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)

    def loss(self,input,real_samps,num_D,if_l1=True,l1_lambda=10.,gp_alpha=10.,fm_lambda=10.):
        loss_G=0
        loss_D=0
        loss_fm=0
        if type(input)==list:
            dis_input=input[-1]
        else:
            dis_input=input

        fake_samps=self.gen(input)
        r_preds = self.dis(torch.cat((real_samps, dis_input), dim=1))
        f_preds = self.dis(torch.cat((fake_samps.detach(), dis_input), dim=1))
        for i in range(num_D):
            real_loss = -torch.mean(r_preds[i][-1])
            fake_loss = torch.mean(f_preds[i][-1])
            loss_D += (real_loss+fake_loss)

        alpha = torch.rand(real_samps.size(0), 1, 1, 1).to(real_samps.device)
        x_hat = (alpha * real_samps.detach() + (1 - alpha) * fake_samps.detach()).requires_grad_(True)
        g_preds = self.dis(torch.cat((x_hat, dis_input),dim=1))
        loss_gp=0
        for i in range(num_D):
            loss_gp += self.gradient_penalty(g_preds[i][-1], x_hat)
        loss_D+=loss_gp*gp_alpha

        preds = self.dis(torch.cat((fake_samps, dis_input),dim=1))
        for i in range(num_D):
            for j in range(len(preds)):
                loss_fm+=self.criterion_fm(preds[i][j],r_preds[i][j].detach())
            loss_G+=-torch.mean(preds[i][-1])
        loss_G+=loss_fm*(1/num_D)*fm_lambda
        if if_l1:
            loss_G+=l1_lambda*self.criterion_pix(fake_samps,real_samps)
        return loss_D,loss_G,loss_fm







