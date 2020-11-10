import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from models import *


def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class ImageCompressor(nn.Module):
    def __init__(self, out_channel_N=192, out_channel_M=320):
        super(ImageCompressor, self).__init__()
        self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.priorDecoder = Synthesis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.GMM_Module = GMM_Module(out_channel_M, k=4)
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_M, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        # print('debug', quant_noise_feature.shape)
        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 64, input_image.size(3) // 64).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
        feature = self.Encoder(input_image)
        batch_size = feature.size()[0]

        z = self.priorEncoder(feature)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
        recon_sigma = self.priorDecoder(compressed_z)
        sigma = self.GMM_Module(recon_sigma)
        feature_renorm = feature
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)
        recon_image = self.Decoder(compressed_feature_renorm)
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))

        def feature_probs_based_sigma(feature, sigma):
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-10, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob
        
        
        def cal_total_bits(probs):
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50)) # equal torch.log2(probs)
            return total_bits
        


        # total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        # likelihoods = self.feature_probs_based_sigma_MoG(compressed_feature_renorm, sigma)
        # likelihoods = self.feature_probs_based_sigma_MoG_k2(compressed_feature_renorm, sigma)
        likelihoods = self.feature_probs_based_sigma_MoG_k4(compressed_feature_renorm, sigma)
        total_bits_feature = cal_total_bits(likelihoods)
        # print(likelihoods.shape)
        
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z
        return clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp

    def feature_probs_based_sigma_MoG(self, feature, GMM_paras):
        """
        based MoG prob compute
        :param feature: shape(batch_size, 16, 16, 5N)
        :param sigma:
        :return:
        """
        # print(GMM_paras.shape)
        GMM_paras = GMM_paras.clamp(1e-10, 1e10)
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = \
            torch.split(GMM_paras, split_size_or_sections=self.out_channel_M, dim=1)
        scale0 = torch.abs(scale0)
        scale1 = torch.abs(scale1)
        scale2 = torch.abs(scale2)
    
        probs = torch.stack([prob0, prob1, prob2], dim=-1)
        probs = F.softmax(probs, dim=-1)  # get GMM weights
    
        # to merge them together
        # means = torch.stack([mean0, mean1, mean2], dim=-1)
        # variances = torch.stack([scale0, scale1, scale2], dim=-1)
    
        # calculate the likelihoods for each inputs symbol
        # gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        dist_0 = torch.distributions.Normal(mean0, scale0)
        dist_1 = torch.distributions.Normal(mean1, scale1)
        dist_2 = torch.distributions.Normal(mean2, scale2)
    
        # ===========Gaussian Mixture Model=====================
        likelihoods_0 = dist_0.cdf(feature + 0.5) - dist_0.cdf(feature - 0.5)
        likelihoods_1 = dist_1.cdf(feature + 0.5) - dist_1.cdf(feature - 0.5)
        likelihoods_2 = dist_2.cdf(feature + 0.5) - dist_2.cdf(feature - 0.5)
    
        likelihoods = probs[:, :, :, :, 0] * likelihoods_0 + probs[:, :, :, :, 1] * likelihoods_1 \
                      + probs[:, :, :, :, 2] * likelihoods_2
    
        # =======REVISION: Robust version ==========
        # edge_min = probs[:, :, :, :, 0] * dist_0.cdf(feature + 0.5) + \
        #            probs[:, :, :, :, 1] * dist_1.cdf(feature + 0.5) + \
        #            probs[:, :, :, :, 2] * dist_2.cdf(feature + 0.5)
        #
        # edge_max = probs[:, :, :, :, 0] * (1.0 - dist_0.cdf(feature - 0.5)) + \
        #            probs[:, :, :, :, 1] * (1.0 - dist_1.cdf(feature - 0.5)) + \
        #            probs[:, :, :, :, 2] * (1.0 - dist_2.cdf(feature - 0.5))
        # likelihoods = torch.where(feature < -254.5, edge_min, torch.where(feature > 255.5, edge_max, likelihoods))
    
        likelihood_lower_bound = torch.tensor(1e-6)
        likelihood_upper_bound = torch.tensor(1.0)
        likelihoods = torch.clamp(likelihoods, likelihood_lower_bound, likelihood_upper_bound)
        # likelihoods = torch.min(torch.max(likelihoods, likelihood_lower_bound), likelihood_upper_bound)
    
        return likelihoods

    def feature_probs_based_sigma_MoG_k2(self, feature, GMM_paras):
        """
        based MoG prob compute
        :param feature: shape(batch_size, 16, 16, 5N)
        :param sigma:
        :return:
        """
        # print(GMM_paras.shape)
        GMM_paras = GMM_paras.clamp(1e-10, 1e10)
        prob0, mean0, scale0, prob1, mean1, scale1 = \
            torch.split(GMM_paras, split_size_or_sections=self.out_channel_M, dim=1)
        scale0 = torch.abs(scale0)
        scale1 = torch.abs(scale1)
        # scale2 = torch.abs(scale2)
    
        probs = torch.stack([prob0, prob1], dim=-1)
        probs = F.softmax(probs, dim=-1)  # get GMM weights
    
        # to merge them together
        # means = torch.stack([mean0, mean1, mean2], dim=-1)
        # variances = torch.stack([scale0, scale1, scale2], dim=-1)
    
        # calculate the likelihoods for each inputs symbol
        # gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        dist_0 = torch.distributions.Normal(mean0, scale0)
        dist_1 = torch.distributions.Normal(mean1, scale1)
        # dist_2 = torch.distributions.Normal(mean2, scale2)
    
        # ===========Gaussian Mixture Model=====================
        likelihoods_0 = dist_0.cdf(feature + 0.5) - dist_0.cdf(feature - 0.5)
        likelihoods_1 = dist_1.cdf(feature + 0.5) - dist_1.cdf(feature - 0.5)
        # likelihoods_2 = dist_2.cdf(feature + 0.5) - dist_2.cdf(feature - 0.5)
    
        likelihoods = probs[:, :, :, :, 0] * likelihoods_0 + probs[:, :, :, :, 1] * likelihoods_1
    
        # =======REVISION: Robust version ==========
        # edge_min = probs[:, :, :, :, 0] * dist_0.cdf(feature + 0.5) + \
        #            probs[:, :, :, :, 1] * dist_1.cdf(feature + 0.5) + \
        #            probs[:, :, :, :, 2] * dist_2.cdf(feature + 0.5)
        #
        # edge_max = probs[:, :, :, :, 0] * (1.0 - dist_0.cdf(feature - 0.5)) + \
        #            probs[:, :, :, :, 1] * (1.0 - dist_1.cdf(feature - 0.5)) + \
        #            probs[:, :, :, :, 2] * (1.0 - dist_2.cdf(feature - 0.5))
        # likelihoods = torch.where(feature < -254.5, edge_min, torch.where(feature > 255.5, edge_max, likelihoods))
    
        likelihood_lower_bound = torch.tensor(1e-6)
        likelihood_upper_bound = torch.tensor(1.0)
        likelihoods = torch.clamp(likelihoods, likelihood_lower_bound, likelihood_upper_bound)
        # likelihoods = torch.min(torch.max(likelihoods, likelihood_lower_bound), likelihood_upper_bound)
    
        return likelihoods

    def feature_probs_based_sigma_MoG_k4(self, feature, GMM_paras):
        """
        based MoG prob compute
        :param feature: shape(batch_size, 16, 16, 5N)
        :param sigma:
        :return:
        """
        # print(GMM_paras.shape)
        GMM_paras = GMM_paras.clamp(1e-10, 1e10)
        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2, prob3, mean3, scale3 = \
            torch.split(GMM_paras, split_size_or_sections=self.out_channel_M, dim=1)
        scale0 = torch.abs(scale0)
        scale1 = torch.abs(scale1)
        scale2 = torch.abs(scale2)
        scale3 = torch.abs(scale3)
    
        probs = torch.stack([prob0, prob1, prob2, prob3], dim=-1)
        probs = F.softmax(probs, dim=-1)  # get GMM weights
    
        # to merge them together
        # means = torch.stack([mean0, mean1, mean2], dim=-1)
        # variances = torch.stack([scale0, scale1, scale2], dim=-1)
    
        # calculate the likelihoods for each inputs symbol
        # gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        dist_0 = torch.distributions.Normal(mean0, scale0)
        dist_1 = torch.distributions.Normal(mean1, scale1)
        dist_2 = torch.distributions.Normal(mean2, scale2)
        dist_3 = torch.distributions.Normal(mean3, scale3)
    
        # ===========Gaussian Mixture Model=====================
        likelihoods_0 = dist_0.cdf(feature + 0.5) - dist_0.cdf(feature - 0.5)
        likelihoods_1 = dist_1.cdf(feature + 0.5) - dist_1.cdf(feature - 0.5)
        likelihoods_2 = dist_2.cdf(feature + 0.5) - dist_2.cdf(feature - 0.5)
        likelihoods_3 = dist_3.cdf(feature + 0.5) - dist_3.cdf(feature - 0.5)
    
        likelihoods = probs[:, :, :, :, 0] * likelihoods_0 + probs[:, :, :, :, 1] * likelihoods_1 \
                      + probs[:, :, :, :, 2] * likelihoods_2 + probs[:, :, :, :, 3] * likelihoods_3
    
        # =======REVISION: Robust version ==========
        # edge_min = probs[:, :, :, :, 0] * dist_0.cdf(feature + 0.5) + \
        #            probs[:, :, :, :, 1] * dist_1.cdf(feature + 0.5) + \
        #            probs[:, :, :, :, 2] * dist_2.cdf(feature + 0.5)
        #
        # edge_max = probs[:, :, :, :, 0] * (1.0 - dist_0.cdf(feature - 0.5)) + \
        #            probs[:, :, :, :, 1] * (1.0 - dist_1.cdf(feature - 0.5)) + \
        #            probs[:, :, :, :, 2] * (1.0 - dist_2.cdf(feature - 0.5))
        # likelihoods = torch.where(feature < -254.5, edge_min, torch.where(feature > 255.5, edge_max, likelihoods))
    
        likelihood_lower_bound = torch.tensor(1e-6)
        likelihood_upper_bound = torch.tensor(1.0)
        likelihoods = torch.clamp(likelihoods, likelihood_lower_bound, likelihood_upper_bound)
        # likelihoods = torch.min(torch.max(likelihoods, likelihood_lower_bound), likelihood_upper_bound)
    
        return likelihoods