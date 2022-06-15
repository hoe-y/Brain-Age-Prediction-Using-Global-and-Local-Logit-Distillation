import torch
from torch import nn
import torch.nn.functional as F
from utils.utils import *

class GLDLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=500.0, spatial_size=24, div=2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_local = div * div 
        self.cross_entropy = nn.MultiLabelSoftMarginLoss()#nn.CrossEntropyLoss()#nn.BCELoss()
        self.mseloss = nn.MSELoss()
        self.t_local_pool = nn.AvgPool3d(kernel_size = (spatial_size  , spatial_size // div, spatial_size//div), stride=(spatial_size // div))
        self.s_local_pool = nn.AvgPool3d(kernel_size = (spatial_size , spatial_size // div, spatial_size//div), stride=(spatial_size // div))
        self.t_global_pool = nn.AvgPool3d(kernel_size=(spatial_size, spatial_size, spatial_size))
        self.s_global_pool = nn.AvgPool3d(kernel_size=(spatial_size, spatial_size, spatial_size))
        self.drop = nn.Dropout(0.5)

    def forward(self, t_f, s_f, t_f1, t_f2, s_f1, s_f2, yy):

        t_global_pen = self.t_global_pool(t_f)
        t_global_pen = t_f1(t_global_pen)
        t_global_pen = t_f2(t_global_pen)
        t_global_pen = t_global_pen.view(t_global_pen.size(0), -1)
        t_global_logit = F.log_softmax(t_global_pen, dim=1)

        t_local_pen = self.t_local_pool(t_f)
        t_local_pen = t_f1(t_local_pen)
        t_local_pen = t_f2(t_local_pen)
        t_local_pen = t_local_pen.view(t_local_pen.size(0), t_local_pen.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        t_local_logit = F.log_softmax(t_local_pen, dim=1)

        s_global_pen = self.s_global_pool(s_f)
        s_global_pen = s_f1(s_global_pen)
        s_global_pen = s_f2(s_global_pen)
        s_global_pen = s_global_pen.view(s_global_pen.size(0), -1)
        s_global_logit = F.log_softmax(s_global_pen, dim=1)

        s_local_pen = self.s_local_pool(s_f)
        s_local_pen = s_f1(s_local_pen)
        s_local_pen = s_f2(s_local_pen)
        s_local_pen = s_local_pen.view(s_local_pen.size(0), s_local_pen.size(1), -1).transpose(dim0=2, dim1=1).flatten(start_dim=0, end_dim=1)
        s_local_logit = F.log_softmax(s_local_pen, dim=1)


        t_logits = torch.cat((t_global_logit, t_local_logit), dim=0)
        s_logits = torch.cat((s_global_logit, s_local_logit), dim=0)
        
        task_loss = my_KLDivLoss(s_global_logit, yy)

        #global_loss = F.kl_div(F.log_softmax(s_global_pen, dim=1), F.softmax(t_global_pen, dim=1),'batchmean')
        #local_loss = F.kl_div(F.log_softmax(s_local_pen, dim=1), F.softmax(t_local_pen, dim=1),'batchmean')

        global_loss = F.kl_div(F.log_softmax(self.mean_var_norm(s_global_pen), dim=1), F.softmax(self.mean_var_norm(t_global_pen), dim=1),'batchmean')
        local_loss = F.kl_div(F.log_softmax(self.mean_var_norm(s_local_pen), dim=1), F.softmax(self.mean_var_norm(t_local_pen), dim=1),'batchmean')

        relation_loss = self.dist_preserve_loss(t_logits, s_logits)
        distill_loss = self.alpha * global_loss + self.num_local * local_loss + self.beta * relation_loss

        return (1. - self.alpha) * task_loss, distill_loss

    def mean_var_norm(self, in_logit):
        norm_output = in_logit / in_logit.std(1).unsqueeze(1)
        return norm_output

    def dist_preserve_loss(self, t, s):
        bsz = s.size()[0]
        f_s = s.view(bsz, -1)
        f_t = t.view(bsz, -1)

        s_square = f_s.pow(2).sum(dim=1)
        s_prod = torch.mm(f_s, torch.t(f_s))

        G_s = (s_square.unsqueeze(1) + s_square.unsqueeze(0) - 2. * s_prod)

        G_s = torch.nn.functional.normalize(G_s)

        t_square = f_t.pow(2).sum(dim=1)
        t_prod = torch.mm(f_t, torch.t(f_t))

        G_t = (t_square.unsqueeze(1) + t_square.unsqueeze(0) - 2. * t_prod)

        G_t = torch.nn.functional.normalize(G_t)

        G_t= torch.nan_to_num(G_t, 1e-8)
        G_diff = G_t - G_s
        
        _sum = (G_diff * G_diff).view(-1, 1)
        _sum = _sum.sum(0)
        loss =  _sum / (bsz * bsz)
        return loss
