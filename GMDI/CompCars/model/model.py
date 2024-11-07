import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model.modules import *
from model.variational_beta import *
import os
from visdom import Visdom
import pickle
import json

from model.lr_scheduler import TransformerLRScheduler
from sklearn.manifold import MDS
from geomloss import SamplesLoss
# const
LARGE_NUM = 1e9


# ========================
def to_np(x):
    return x.detach().cpu().numpy()

def flat_1(x):
    n, m = x.shape[:2]
    return x.reshape(n * m, *x.shape[2:])

def flat(x):
    k, n, m = x.shape[:3]
    return x.reshape(k, n * m, *x.shape[3:])

def write_pickle(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)


# =========================
# the base model
class BaseModel(nn.Module):

    def __init__(self, opt):
        super(BaseModel, self).__init__()
        # set output format
        np.set_printoptions(suppress=True, precision=6)

        self.opt = opt
        self.device = opt.device
        self.use_visdom = opt.use_visdom
        if opt.use_visdom:
            self.env = Visdom(port=opt.visdom_port)
            self.test_pane = dict()
            self.test_pane_init = False

        self.num_domain = opt.num_domain

        self.k = opt.k

        self.outf = self.opt.outf
        self.train_log = self.outf + "/loss.log"
        self.model_path = self.outf + '/model.pth'
        if not os.path.exists(self.opt.outf):
            os.mkdir(self.opt.outf)
        with open(self.train_log, 'w') as f:
            f.write("log start!\n")

        # save all the config to json file
        with open("{}/config.json".format(self.outf), "w") as outfile:
            json.dump(self.opt, outfile, indent=2)

        mask_list = np.zeros(opt.num_domain)
        mask_list[opt.src_domain_idx] = 1
        self.domain_mask = torch.IntTensor(mask_list).to(opt.device)

        # nan flag
        self.nan_flag = False
        
        # var approximation of beta
        if self.k > 1:

            self.beta = var_approx_beta(opt)
        else:
            self.beta = None
        
        # theta:
        self.use_theta_seq = None

        self.init_test = False

    def learn(self, epoch, dataloader):
        self.train()

        self.epoch = epoch
        loss_values = {loss: 0 for loss in self.loss_names}
        self.new_u = []
        
        # init the posterior of beta
        if self.epoch == 0:
            if self.beta is not None:
                eta = torch.zeros(self.num_domain, self.k, device=self.device, dtype=torch.float)
                for i in range(eta.shape[0]):
                    eta[i][i % self.k] = 1.0
                self.beta.update_posterior(eta)

        count = 0
        for data in dataloader.get_data():
            count += 1
            self.__set_input__(data)
            self.__train_forward__()
            new_loss_values = self.__optimize__()

            # for the loss visualization
            for key, loss in new_loss_values.items():
                loss_values[key] += loss

            self.new_u.append(self.u_seq)

        self.new_u = self.my_cat(self.new_u)
        self.use_theta_seq = self.generate_theta(self.new_u)
        
        for key, _ in new_loss_values.items():
            loss_values[key] /= count

        if self.use_visdom:
            self.__vis_loss__(loss_values)

        if (self.epoch + 1) % 10 == 0 or self.epoch == 0:
            print("epoch {}, loss: {}, lambda gan: {}".format(
                self.epoch, loss_values, self.opt.lambda_gan))

        # learning rate decay
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()

        # check nan
        if any(np.isnan(val) for val in loss_values.values()):
            self.nan_flag = True
        else:
            self.nan_flag = False

        # updata the posterior and the prior of beta
        if self.beta is not None:
            self.beta.update_posterior(self.eta_pre.T)
            self.beta.update_prior()

    def test(self, epoch, dataloader):
        # Assuming test on all domains!
        self.eval()
        self.epoch = epoch

        # init the sample number:
        # self.init_test = False
        if not self.init_test:
            # drop last
            batch_num = np.floor(dataloader.sudo_len / self.opt.batch_size)
            factor = np.ceil(self.opt.save_sample / batch_num)
            self.save_sample = int(factor * batch_num)
            self.factor = int(factor)
            self.save_sample_idx = np.arange(self.factor)
            self.init_test = True

        acc_num = torch.zeros(self.num_domain).to(self.device)
        l_x = torch.zeros(self.num_domain, self.save_sample,
                          self.opt.input_dim).to(self.device)
        l_y = torch.zeros(self.num_domain, self.save_sample).to(self.device)
        # l_r_x = []
        l_domain = torch.zeros(self.num_domain,
                               self.save_sample).to(self.device)
        l_label = torch.zeros(self.num_domain,
                              self.save_sample).to(self.device)
        l_encode = torch.zeros(self.k, self.num_domain, self.save_sample,
                               self.opt.num_hidden).to(self.device) 
        l_u = torch.zeros(self.num_domain, self.opt.u_dim).to(self.device)
        l_u_all = torch.zeros(self.num_domain, self.save_sample,
                              self.opt.u_dim).to(self.device)

        sample_count = 0
        # sample a few data points for visualization
        count = 0
        for data in dataloader.get_data():
            self.__set_input__(data)

            # forward
            with torch.no_grad():
                self.__test_forward__()
                # drop last batch
                if self.tmp_batch_size < self.opt.batch_size:
                    continue
                sample_count += self.y_seq.shape[1]
                count += 1
                acc_num += self.g_seq.eq(self.y_seq).to(torch.float).sum(-1)
                l_x[:, (count - 1) * self.factor:count *
                    self.factor, :] = self.x_seq[:, self.save_sample_idx, :]
                l_y[:, (count - 1) * self.factor:count *
                    self.factor] = self.y_seq[:, self.save_sample_idx]
                l_domain[:, (count - 1) * self.factor:count *
                         self.factor] = self.domain_seq[:,
                                                        self.save_sample_idx]
                l_encode[:, :, (count - 1) * self.factor:count *
                         self.factor, :] = self.q_z_seq[:, :, self.
                                                        save_sample_idx, :]
                l_label[:, (count - 1) * self.factor:count *
                        self.factor] = self.g_seq[:, self.save_sample_idx]
                l_u += self.u_seq.sum(1)
                l_u_all[:, (count - 1) * self.factor:count *
                        self.factor, :] = self.u_seq[:,
                                                     self.save_sample_idx, :]

        acc = to_np(acc_num / sample_count)
        test_acc = acc[self.opt.tgt_domain_idx].sum() / (
            self.opt.num_target) * 100
        acc_msg = '[Test][{}] Accuracy: total average {:.1f}, test average {:.1f}, in each domain {}'.format(
            epoch,
            acc.mean() * 100, test_acc, np.around(acc * 100, decimals=1))
        self.__log_write__(acc_msg)
        if self.use_visdom:
            self.__vis_test_error__(test_acc, 'test acc')

        d_all = dict()
        d_all['acc_msg'] = acc_msg
        d_all['data'] = flat_1(to_np(l_x))
        d_all['gt_label'] = flat_1(to_np(l_y))
        d_all['domain'] = flat_1(to_np(l_domain))
        d_all['label'] = flat_1(to_np(l_label))
        d_all['encodeing'] = flat(to_np(l_encode))
        d_all['u'] = to_np(l_u / self.save_sample)
        # d_all['r_x'] = flat(r_x_all)
        d_all['u_all'] = flat_1(to_np(l_u_all))
        d_all['theta'] = to_np(self.theta_seq)
        d_all['z'] = to_np(self.q_z_seq)

        if (
                self.epoch + 1
        ) % self.opt.save_interval == 0 or self.epoch + 1 == self.opt.num_epoch:
            write_pickle(d_all, self.opt.outf + '/' + str(epoch) + '_pred.pkl')

        return test_acc, self.nan_flag
    
    def inference(self, dataloader):
        self.test(epoch=self.opt.num_epoch-1, dataloader=dataloader)

    def my_cat(self, new_u_seq):
        # concatenation of local domain index u
        st = new_u_seq[0]
        idx_end = len(new_u_seq)
        for i in range(1, idx_end):
            st = torch.cat((st, new_u_seq[i]), dim=1)
        return st

    def __vis_test_error__(self, loss, title):
        if not self.test_pane_init:
            self.test_pane[title] = self.env.line(X=np.array([self.epoch]),
                                                  Y=np.array([loss]),
                                                  opts=dict(title=title))
            self.test_pane_init = True
        else:
            self.env.line(X=np.array([self.epoch]),
                          Y=np.array([loss]),
                          win=self.test_pane[title],
                          update='append')

    def save(self):
        torch.save(self.netU.state_dict(), self.outf + '/netU.pth')
        torch.save(self.netUCon.state_dict(), self.outf + '/netUCon.pth')
        torch.save(self.netZ.state_dict(), self.outf + '/netZ.pth')
        torch.save(self.netF.state_dict(), self.outf + '/netF.pth')
        torch.save(self.netR.state_dict(), self.outf + '/netR.pth')
        torch.save(self.netD.state_dict(), self.outf + '/netD.pth')
        torch.save(self.netTheta.state_dict(), self.outf + '/netTheta.pth')
        torch.save(self.netTheta2U.state_dict(), self.outf + '/netTheta2U.pth')
        if self.beta is not None:
            torch.save(self.beta.var_gamma1, self.outf + '/var_gamma1.pth')
            torch.save(self.beta.var_gamma2, self.outf + '/var_gamma2.pth')        

    def __set_input__(self, data, train=True):

        x_seq, y_seq, domain_seq = [d[0][None, :, :] for d in data
                                    ], [d[1][None, :] for d in data
                                        ], [d[2][None, :] for d in data]

        self.x_seq = torch.cat(x_seq, 0).to(self.device)
        self.y_seq = torch.cat(y_seq, 0).to(self.device)
        self.domain_seq = torch.cat(domain_seq, 0).to(self.device)
        self.tmp_batch_size = self.x_seq.shape[1]

    def __train_forward__(self):
        self.u_seq, self.u_mu_seq, self.u_log_var_seq = self.netU(self.x_seq)

        self.u_con_seq = self.netUCon(self.u_seq)

        if self.use_theta_seq != None:
            self.theta_seq, self.theta_log_var_seq = self.netTheta(
                self.use_theta_seq, self.use_theta_seq)
        else:
            self.tmp_theta_seq = self.generate_theta(self.u_seq)
            self.theta_seq, self.theta_log_var_seq = self.netTheta(
                self.tmp_theta_seq, self.tmp_theta_seq)

        self.theta_U_seq = self.netTheta2U(self.theta_seq)

        self.q_z_seq, self.q_z_mu_seq, self.q_z_log_var_seq, self.p_z_seq, self.p_z_mu_seq, self.p_z_log_var_seq, = self.netZ(
            self.x_seq, self.u_seq, self.theta_seq)
        
        self.r_x_seq = self.netR(self.u_seq)
        self.f_seq = self.netF(self.q_z_seq)

        self.d_seq = self.netD(self.q_z_seq)
        self.loss_D = self.__loss_D__(self.d_seq)

    def __test_forward__(self):
        self.u_seq, self.u_mu_seq, self.u_log_var_seq = self.netU(self.x_seq)
        self.u_con_seq = self.netUCon(self.u_seq)
        if self.use_theta_seq != None:
            self.theta_seq, self.theta_log_var_seq = self.netTheta(self.use_theta_seq,
                                            self.use_theta_seq)
        else:
            self.tmp_theta_seq = self.generate_theta(self.u_seq)
            self.theta_seq, self.theta_log_var_seq = self.netTheta(self.tmp_theta_seq,
                                            self.tmp_theta_seq)

        self.q_z_seq, self.q_z_mu_seq, self.q_z_log_var_seq, self.p_z_seq, self.p_z_mu_seq, self.p_z_log_var_seq, = self.netZ(
            self.x_seq, self.u_seq, self.theta_seq)
        self.f_seq = self.netF(self.q_z_seq)

        
        self.theta_U_seq = self.netTheta2U(self.theta_seq)
        theta_t = self.theta_U_seq.unsqueeze(dim=2).expand(self.k, -1, self.tmp_batch_size, -1)

        loss_p_u_theta = ((self.u_seq - theta_t)**2).sum(3)
        loss_p_u_theta = -torch.mean(loss_p_u_theta, dim=2)

        var_theta = torch.exp(self.theta_log_var_seq) 
        loss_theta = -torch.mean((var_theta).sum(dim=2), dim=1)

        vk_likelihood = self.get_vk_likelihood()
        
        loss_p_u_theta = torch.stack([x.view(-1).to(device=self.device) for x in loss_p_u_theta], dim=0) 
        loss_theta = torch.stack([x.view(-1).to(device=self.device) for x in loss_theta], dim=0)

        ll_vk = torch.stack([x.view(-1).to(device=self.device) for x in vk_likelihood]) 

        # caculate the posterior of v
        pos_v = torch.exp(ll_vk-loss_theta.mean(1).unsqueeze(1).expand(self.k, self.num_domain)+loss_p_u_theta.mean(1).unsqueeze(1).expand(self.k, self.num_domain) - torch.logsumexp(ll_vk-loss_theta.mean(1).unsqueeze(1).expand(self.k, self.num_domain)+loss_p_u_theta.mean(1).unsqueeze(1).expand(self.k, self.num_domain), dim=0)) 

        self.g_seq_list = torch.zeros_like(self.f_seq[0], device=self.device)
        for i in range(self.k):
            self.g_seq_list += pos_v[i].unsqueeze(1).unsqueeze(2) * self.f_seq[i]
        self.g_seq = torch.argmax(self.g_seq_list, dim=2)

        


    def __optimize__(self):
        loss_value = dict()
        loss_value['D'], loss_value['E_pred'], loss_value['Q_u_x'], loss_value['Q_z_x_u'], loss_value['P_z_x_u'], \
            loss_value['U_concentrate'], loss_value['R'], loss_value['U_theta_R'], loss_value['P_theta'] \
                = self.__optimize_DUZF__()

        return loss_value

    def contrastive_loss(self, u_con_seq, temperature=1):
        u_con_seq = u_con_seq.reshape(self.tmp_batch_size * self.num_domain,
                                      -1)
        u_con_seq = nn.functional.normalize(u_con_seq, p=2, dim=1)

        # calculate the cosine similarity between each pair
        logits = torch.matmul(u_con_seq, torch.t(u_con_seq)) / temperature

        # we only choose the one that is:
        # 1, belongs to one domain
        # 2, next to each other
        # as the pair that we want to concentrate them, and all the others will be cancel out

        # the first 2 steps will generate matrix in this format:
        # [0, 1, 0, 0]
        # [0, 0, 1, 0]
        # [0, 0, 0, 1]
        # [1, 0, 0, 0]
        base_m = torch.diag(torch.ones(self.tmp_batch_size - 1),
                            diagonal=1).to(self.device)
        base_m[self.tmp_batch_size - 1, 0] = 1

        # Then we generate the "complementary" matrix in this format:
        # [1, 0, 1, 1]
        # [1, 1, 0, 1]
        # [1, 1, 1, 0]
        # [0, 1, 1, 1]
        # which will be used in the mask
        base_m = torch.ones(self.tmp_batch_size, self.tmp_batch_size).to(
            self.device) - base_m
        # generate the true mask with the base matrix as block.
        # [1, 0, 1, 1, 0, 0, 0, 0 ...]
        # [1, 1, 0, 1, 0, 0, 0, 0 ...]
        # [1, 1, 1, 0, 0, 0, 0, 0 ...]
        # [0, 1, 1, 1, 0, 0, 0, 0 ...]
        # [0, 0, 0, 0, 1, 0, 1, 1 ...]
        # [0, 0, 0, 0, 1, 1, 0, 1 ...]
        # [0, 0, 0, 0, 1, 1, 1, 0 ...]
        # [0, 0, 0, 0, 0, 1, 1, 1 ...]
        masks = torch.block_diag(*([base_m] * self.num_domain))
        logits = logits - masks * LARGE_NUM

        # label: which similarity should maximize. We only maximize the similarity of datapoints that:
        # belongs to one domain
        # next to each other
        label = torch.arange(self.tmp_batch_size * self.num_domain).to(
            self.device)
        label = torch.remainder(label + 1, self.tmp_batch_size) + label.div(
            self.tmp_batch_size, rounding_mode='floor') * self.tmp_batch_size
        
        loss_u_concentrate = F.cross_entropy(logits, label)
        return loss_u_concentrate

    def __optimize_DUZF__(self):
        self.train()

        self.optimizer_UZF.zero_grad()

        # - E_q[log q(u|x)]
        loss_q_u_x = torch.mean((0.5 * self.u_log_var_seq).sum(2), dim=1)

        # - E_q[log q(z|x,u)]
        loss_q_z_x_u = torch.mean((0.5 * self.q_z_log_var_seq).sum(3),
                                  dim=2)
        
        # E_q[log p(z|x,u)]
        # first one is for normal
        loss_p_z_x_u = -0.5 * self.p_z_log_var_seq - 0.5 * (
            torch.exp(self.q_z_log_var_seq) +
            (self.q_z_mu_seq - self.p_z_mu_seq)**2) / torch.exp(self.p_z_log_var_seq)
        loss_p_z_x_u = torch.mean(loss_p_z_x_u.sum(3), dim=2)      

        # E_q[log p(y|z)]
        y_seq_source = self.y_seq[self.domain_mask == 1]
        loss_p_y_z = []
        for _, f_seq_temp in enumerate(self.f_seq):
            f_seq_source = f_seq_temp[self.domain_mask == 1]
            loss_p_y_z_temp = -F.nll_loss(
            flat_1(f_seq_source).squeeze(), flat_1(y_seq_source))
            loss_p_y_z.append(loss_p_y_z_temp)

        # E_q[log p(\theta)]
        var_theta = torch.exp(self.theta_log_var_seq) 
        loss_theta = -torch.sum(var_theta, dim=2)

        # loss for u and theta
        # log p(u|theta)        
        theta_t = self.theta_U_seq.unsqueeze(dim=2).expand(self.k, 
            -1, self.tmp_batch_size, -1)
        loss_p_u_theta = ((self.u_seq - theta_t)**2).sum(3)
        loss_p_u_theta = -torch.mean(loss_p_u_theta, dim=2)

        # concentrate loss
        loss_u_concentrate = self.contrastive_loss(self.u_con_seq)

        # reconstruction loss (p(x|u))
        loss_p_x_u = ((self.x_seq - self.r_x_seq)**2).sum(2)
        loss_p_x_u = -torch.mean(loss_p_x_u, dim=1)

        # gan loss (adversarial loss)
        if self.opt.lambda_gan != 0:
            if self.opt.d_loss_type == "ADDA_loss":
                loss_E_gan = []
                for _, d_seq_temp in enumerate(self.d_seq):
                    d_seq_target = d_seq_temp[self.domain_mask == 0]

                    loss_E_gan_temp = -torch.log(d_seq_target + 1e-10).mean([1,2])
                    loss_E_gan.append(loss_E_gan_temp)
                
            else:
                loss_E_gan = -self.loss_D
        else:
            loss_E_gan = torch.tensor(0, dtype=torch.double, device=self.opt.device)
        

        vk_likelihood = self.get_vk_likelihood()

        loss_E_gan = torch.stack([x.view(-1).to(device=self.device) for x in loss_E_gan], dim=0)
        loss_u_concentrate_11 = loss_u_concentrate.expand(self.k, self.num_domain)
        loss_u_concentrate = torch.stack([x.view(-1).to(device=self.device) for x in loss_u_concentrate_11], dim=0)
        loss_p_y_z = torch.stack([x.view(-1).to(device=self.device) for x in loss_p_y_z], dim=0)
        loss_p_u_theta = torch.stack([x.view(-1).to(device=self.device) for x in loss_p_u_theta], dim=0)
        loss_p_z_x_u = torch.stack([x.view(-1).to(device=self.device) for x in loss_p_z_x_u], dim=0)
        loss_q_z_x_u = torch.stack([x.view(-1).to(device=self.device) for x in loss_q_z_x_u], dim=0)
        loss_theta = torch.stack([x.view(-1).to(device=self.device) for x in loss_theta], dim=0)

        self.loss_D = torch.cat([x.view(-1).to(device=self.device) for x in self.loss_D], dim=0)

        ll_vk = torch.stack([x.view(-1).to(device=self.device) for x in vk_likelihood])

        # calculate posterior of v
        eta = torch.exp(ll_vk-loss_theta.mean(1).unsqueeze(1).expand(self.k, self.num_domain)+loss_p_u_theta.mean(1).unsqueeze(1).expand(self.k, self.num_domain) - torch.logsumexp(ll_vk-loss_theta.mean(1).unsqueeze(1).expand(self.k, self.num_domain)+loss_p_u_theta.mean(1).unsqueeze(1).expand(self.k, self.num_domain), dim=0))

        self.eta_pre = eta.detach()

        loss_E_gan = torch.mean((eta.T[self.domain_mask == 0]).T * loss_E_gan.mean(1).view(-1, 1), dim=1).sum(0)
        loss_u_concentrate = torch.mean(eta * loss_u_concentrate.mean(1).view(-1, 1), dim=1).sum(0)
        loss_p_x_u = torch.mean(eta.sum(0) * loss_p_x_u)
        loss_p_u_theta = torch.mean(eta * loss_p_u_theta.mean(1).view(-1, 1), dim=1).sum(0)
        loss_theta = torch.mean(eta * loss_theta.mean(1).view(-1, 1), dim=1).sum(0)
        loss_p_y_z = torch.mean((eta.T[self.domain_mask == 1]).T *  loss_p_y_z.mean(1).view(-1, 1), dim=1).sum(0)
        loss_q_u_x = torch.mean(eta.sum(0) * loss_q_u_x)
        loss_q_z_x_u = torch.mean(eta *  loss_q_z_x_u.mean(1).view(-1, 1), dim=1).sum(0)
        loss_p_z_x_u = torch.mean(eta * loss_p_z_x_u.mean(1).view(-1, 1), dim=1).sum(0)

        self.loss_D = torch.sum(eta.mean(1) * self.loss_D)
        
        loss_E = loss_E_gan * self.opt.lambda_gan + self.opt.lambda_u_concentrate * loss_u_concentrate - (
            self.opt.lambda_reconstruct * loss_p_x_u + self.opt.lambda_u_theta *
            loss_p_u_theta + self.opt.lambda_theta * loss_theta +
            loss_p_y_z + loss_q_u_x + loss_q_z_x_u + loss_p_z_x_u)

        self.optimizer_D.zero_grad()
        self.loss_D.backward(retain_graph=True)
        self.optimizer_UZF.zero_grad()
        loss_E.backward()

        self.optimizer_D.step()
        self.optimizer_UZF.step()

        return self.loss_D.item(), -loss_p_y_z.item(), loss_q_u_x.item(
        ), loss_q_z_x_u.item(), loss_p_z_x_u.item(), loss_u_concentrate.item(
        ), -loss_p_x_u.item(), -loss_p_u_theta.item(), -loss_theta.item()
    
    def get_vk_likelihood(self):
        vk_likelihood_list = []
        for kth in range(self.k):
            kl_layer_ls = []
            for mu_p, sig_p, mu_q, sig_q in zip(self.p_z_mu_seq[kth], torch.exp(self.p_z_log_var_seq[kth]), self.q_z_mu_seq[kth], torch.exp(self.q_z_log_var_seq[kth])):
                mean_diff = mu_q - mu_p
                sig_q_inv = 1 / sig_q
                kl_layer = torch.log(sig_q).sum(dim=1) - torch.log(sig_p).sum(dim=1) - mu_p[0].numel() + (sig_q_inv * sig_p).sum(dim=1) \
                        + ((mean_diff * sig_q_inv) * mean_diff).sum(dim=1)
                kl_layer_ls.append(kl_layer)
            # 0.01 is the down-wighting hyperparameter of KL-divergence, which is used  to avoid over-fitting.
            kl = 0.01 * torch.stack([kl_layer.sum() for kl_layer in kl_layer_ls]) / 2 
            exp_pi = torch.ones(self.num_domain, device=self.device, dtype=torch.float)
            for i in range(kth):
                exp_pi = exp_pi * (1 - self.beta.var_gamma1[:, i].to(device=self.device) / \
                        (self.beta.var_gamma1[:, i].to(device=self.device) + self.beta.var_gamma2[:, i].to(device=self.device)))
            if kth < self.k - 1:
                exp_pi = exp_pi * self.beta.var_gamma1[:, kth].to(device=self.device) / \
                        (self.beta.var_gamma1[:, kth].to(device=self.device) + self.beta.var_gamma2[:, kth].to(device=self.device))                
            vk_likelihood = exp_pi - kl
            vk_likelihood_list.append(vk_likelihood)
        return vk_likelihood_list
    
    def __log_write__(self, loss_msg):
        print(loss_msg)
        with open(self.train_log, 'a') as f:
            f.write(loss_msg + "\n")

    def __vis_loss__(self, loss_values):
        if self.epoch == 0:
            self.panes = {
                loss_name: self.env.line(
                    X=np.array([self.epoch]),
                    Y=np.array([loss_values[loss_name]]),
                    opts=dict(title='loss for {} on epochs'.format(loss_name)))
                for loss_name in self.loss_names
            }
        else:
            for loss_name in self.loss_names:
                self.env.line(X=np.array([self.epoch]),
                              Y=np.array([loss_values[loss_name]]),
                              win=self.panes[loss_name],
                              update='append')

    def __init_weight__(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                # print("init linear weight!")
                nn.init.normal_(m.weight, mean=0, std=0.01)
                #                 nn.init.normal_(m.weight, mean=0, std=0.1)
                #                 nn.init.xavier_normal_(m.weight, gain=10)
                nn.init.constant_(m.bias, val=0)


class VDI(BaseModel):
    #########
    # VDI (Variational Domain Index) Model
    #########

    def __init__(self, opt, search_space=None):
        super(VDI, self).__init__(opt)

        self.bayesian_opt = False
        if search_space != None:
            self.bayesian_opt = True

        self.netU = UNet(opt).to(opt.device)
        self.netUCon = UConcenNet(opt).to(opt.device)
        self.netZ = Q_ZNet_theta(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device)
        self.netR = ReconstructNet(opt).to(opt.device)

        self.netTheta = ThetaNet(opt).to(opt.device).float()
        self.netTheta2U = Theta2UNet(opt).to(opt.device).float()

        # for DANN-style discriminator loss & MDS aggregation
        if self.opt.d_loss_type == "DANN_loss":
            self.netD = ClassDiscNet(opt).to(opt.device)
            self.__loss_D__ = self.__loss_D_dann__
            self.generate_theta = self.__reconstruct_u_graph__
        # for DANN-style discriminator loss & mean aggregation
        elif self.opt.d_loss_type == "DANN_loss_mean":
            assert self.opt.u_dim == self.opt.theta_dim, "When you use \"mean\" as aggregation, you should make sure local domain index and global domain index have the same dimension."
            self.netD = ClassDiscNet(opt).to(opt.device)
            self.__loss_D__ = self.__loss_D_dann__
            self.generate_theta = self.__u_mean__
            self.netTheta2U = nn.Identity().to(opt.device)
        # for ADDA-style discriminator loss & MDS aggregation
        elif self.opt.d_loss_type == "ADDA_loss":
            self.netD = DiscNet(opt).to(opt.device)
            self.__loss_D__ = self.__loss_D_adda__
            # self.generate_theta = self.__u_mean__
            self.generate_theta = self.__reconstruct_u_graph__
        # for CIDA-style discriminator loss & MDS aggregation
        elif self.opt.d_loss_type == "CIDA_loss":
            self.netD = DiscNet(opt).to(opt.device)
            self.__loss_D__ = self.__loss_D_cida__
            self.generate_theta = self.__u_mean__
        # for GRDA-style discriminator & MDS aggregation
        elif self.opt.d_loss_type == "GRDA_loss":
            self.netD = GraphDNet(opt).to(opt.device)
            self.__loss_D__ = self.__loss_D_grda__
            self.generate_theta = self.__reconstruct_u_graph__
        self.__init_weight__()

        if self.opt.use_pretrain_R:
            pretrain_model_U = torch.load(self.opt.pretrain_U_path)
            pretrain_model_R = torch.load(self.opt.pretrain_R_path)

            self.netU.load_state_dict(pretrain_model_U)
            self.netR.load_state_dict(pretrain_model_R)

        if self.opt.use_pretrain_model_all:
            self.netU.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netU.pth'))
            self.netUCon.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netUCon.pth'))
            self.netZ.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netZ.pth'))
            self.netF.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netF.pth'))
            self.netR.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netR.pth'))
            self.netD.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netD.pth'))
            self.netTheta.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netTheta.pth'))
            self.netTheta2U.load_state_dict(torch.load(self.opt.pretrain_model_all_path + '/netTheta2U.pth'))
            self.beta.var_gamma1 = torch.load(self.opt.pretrain_model_all_path + '/var_gamma1.pth')
            self.beta.var_gamma2 = torch.load(self.opt.pretrain_model_all_path + '/var_gamma2.pth')

        if self.opt.fix_u_r:
            UZF_parameters = list(self.netZ.parameters()) + list(
                self.netF.parameters())
        else:
            UZF_parameters = list(self.netU.parameters()) + list(
                self.netZ.parameters()) + list(self.netF.parameters()) + list(
                    self.netR.parameters()) + list(self.netUCon.parameters())

        UZF_parameters += list(self.netTheta.parameters()) + list(
            self.netTheta2U.parameters())

        self.optimizer_UZF = optim.Adam(UZF_parameters,
                                        lr=opt.init_lr,
                                        betas=(opt.theta1, 0.999))
        self.optimizer_D = optim.Adam(self.netD.parameters(),
                                      lr=opt.init_lr,
                                      betas=(opt.theta1, 0.999))
        self.lr_scheduler_UZF = TransformerLRScheduler(
            optimizer=self.optimizer_UZF,
            init_lr=opt.init_lr,
            peak_lr=opt.peak_lr_e,
            warmup_steps=opt.warmup_steps,
            decay_steps=opt.num_epoch - opt.warmup_steps,
            gamma=0.5**(1 / 100),
            final_lr=opt.final_lr)
        self.lr_scheduler_D = TransformerLRScheduler(
            optimizer=self.optimizer_D,
            init_lr=opt.init_lr,
            peak_lr=opt.peak_lr_d,
            warmup_steps=opt.warmup_steps,
            decay_steps=opt.num_epoch - opt.warmup_steps,
            gamma=0.5**(1 / 100),
            final_lr=opt.final_lr)

        self.lr_schedulers = [self.lr_scheduler_UZF, self.lr_scheduler_D]
        self.loss_names = [
            'D', 'E_pred', 'Q_u_x', 'Q_z_x_u', 'P_z_x_u', 'U_theta_R',
            'U_concentrate', 'R', 'P_theta'
        ]

        # for mds(u)
        self.embedding = MDS(n_components=self.opt.theta_dim,
                             dissimilarity='precomputed')

    def __u_mean__(self, u_seq):
        mu_theta = u_seq.mean(1).detach()
        mu_theta_mean = mu_theta.mean(0, keepdim=True)
        mu_theta_std = mu_theta.std(0, keepdim=True)
        mu_theta_std = torch.maximum(mu_theta_std,
                                    torch.ones_like(mu_theta_std) * 1e-12)
        mu_theta = (mu_theta - mu_theta_mean) / mu_theta_std
        return mu_theta

    def __reconstruct_u_graph__(self, u_seq):
        with torch.no_grad():
            A = torch.zeros(self.num_domain, self.num_domain)
            new_u = u_seq.detach()
            # ~ Wasserstein Loss
            loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            for i in range(self.num_domain):
                for j in range(i + 1, self.num_domain):
                    A[i][j] = loss(new_u[i], new_u[j])
                    A[j][i] = A[i][j]

            A_np = to_np(A)
            bound = np.sort(A.flatten())[int(self.num_domain**2 * 1 / 4)]
            # generate self.A
            self.A = (A_np < bound)

            # calculate the theta seq
            mu_theta = self.embedding.fit_transform(A_np)
            mu_theta = torch.from_numpy(mu_theta).to(self.device)
            # new normalization:
            mu_theta_mean = mu_theta.mean(0, keepdim=True)
            mu_theta_std = mu_theta.std(0, keepdim=True)
            mu_theta_std = torch.maximum(mu_theta_std,
                                        torch.ones_like(mu_theta_std) * 1e-12)
            mu_theta = (mu_theta - mu_theta_mean) / mu_theta_std

            return mu_theta

    def __loss_D_dann__(self, d_seq):
        # this is for DANN
        return F.nll_loss(flat(d_seq),
                          flat(self.domain_seq))  # , self.u_seq.mean(1)

    def __loss_D_adda__(self, d_seqs):
        loss_D_list = []
        for _, d_seq in enumerate(d_seqs):
            d_seq_source = d_seq[self.domain_mask == 1]
            d_seq_target = d_seq[self.domain_mask == 0]
            # D: discriminator loss from classifying source v.s. target
            loss_D = (-torch.log(d_seq_source + 1e-10).mean() -
                    torch.log(1 - d_seq_target + 1e-10).mean())
            loss_D_list.append(loss_D)
        return loss_D_list

    def __loss_D_cida__(self, d_seq):
        # this is for CIDA
        # use L1 instead of L2
        return F.l1_loss(flat(d_seq),
                         flat(self.u_seq.detach()))  # , self.u_seq.mean(1)

    def __loss_D_grda__(self, d_seq):
        # this is for GRDA
        A = self.A

        criterion = nn.BCEWithLogitsLoss()
        d = d_seq
        # random pick subchain and optimize the D
        # balance coefficient is calculate by pos/neg ratio
        # A is the adjancency matrix
        sub_graph = self.__sub_graph__(my_sample_v=self.opt.sample_v, A=A)

        errorD_connected = torch.zeros((1, )).to(self.device)  # .double()
        errorD_disconnected = torch.zeros((1, )).to(self.device)  # .double()

        count_connected = 0
        count_disconnected = 0

        for i in range(self.opt.sample_v):
            v_i = sub_graph[i]
            # no self loop version!!
            for j in range(i + 1, self.opt.sample_v):
                v_j = sub_graph[j]
                label = torch.full(
                    (self.tmp_batch_size, ),
                    A[v_i][v_j],
                    device=self.device,
                )
                # dot product
                if v_i == v_j:
                    idx = torch.randperm(self.tmp_batch_size)
                    output = (d[v_i][idx] * d[v_j]).sum(1)
                else:
                    output = (d[v_i] * d[v_j]).sum(1)

                if A[v_i][v_j]:  # connected
                    errorD_connected += criterion(output, label)
                    count_connected += 1
                else:
                    errorD_disconnected += criterion(output, label)
                    count_disconnected += 1

        # prevent nan
        if count_connected == 0:
            count_connected = 1
        if count_disconnected == 0:
            count_disconnected = 1

        errorD = 0.5 * (errorD_connected / count_connected +
                        errorD_disconnected / count_disconnected)
        # this is a loss balance
        return errorD * self.num_domain

    def __sub_graph__(self, my_sample_v, A):
        # sub graph tool for grda loss
        if np.random.randint(0, 2) == 0:
            return np.random.choice(self.num_domain,
                                    size=my_sample_v,
                                    replace=False)

        # subsample a chain (or multiple chains in graph)
        left_nodes = my_sample_v
        choosen_node = []
        vis = np.zeros(self.num_domain)
        while left_nodes > 0:
            chain_node, node_num = self.__rand_walk__(vis, left_nodes, A)
            choosen_node.extend(chain_node)
            left_nodes -= node_num

        return choosen_node

    def __rand_walk__(self, vis, left_nodes, A):
        # graph random sampling tool for grda loss
        chain_node = []
        node_num = 0
        # choose node
        node_index = np.where(vis == 0)[0]
        st = np.random.choice(node_index)
        vis[st] = 1
        chain_node.append(st)
        left_nodes -= 1
        node_num += 1

        cur_node = st
        while left_nodes > 0:
            nx_node = -1
            node_to_choose = np.where(vis == 0)[0]
            num = node_to_choose.shape[0]
            node_to_choose = np.random.choice(node_to_choose,
                                              num,
                                              replace=False)

            for i in node_to_choose:
                if cur_node != i:
                    # have an edge and doesn't visit
                    if A[cur_node][i] and not vis[i]:
                        nx_node = i
                        vis[nx_node] = 1
                        chain_node.append(nx_node)
                        left_nodes -= 1
                        node_num += 1
                        break
            if nx_node >= 0:
                cur_node = nx_node
            else:
                break
        return chain_node, node_num
