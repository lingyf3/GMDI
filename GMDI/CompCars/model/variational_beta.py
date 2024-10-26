import numpy as np
import torch

class var_approx_beta():

    def __init__(self, opt):
        super(var_approx_beta, self).__init__()
        
        alpha = opt.alpha
        self.k = opt.k
        self.num_domain = opt.num_domain
        device = opt.device
        if isinstance(alpha, int):
            self.prior_alpha1 = torch.ones(self.num_domain, self.k-1, device=device, dtype=torch.float)
            self.prior_alpha2 = torch.full((self.num_domain, self.k-1), float(alpha), device=device, dtype=torch.float)
        else:
            assert alpha.shape[1] == self.k-1
            self.prior_alpha1 = torch.tensor(alpha[0], device=device).float()
            self.prior_alpha2 = torch.tensor(alpha[1], device=device).float()

    
    def update_posterior(self, eta):

        self.var_gamma1 = self.prior_alpha1 +  eta[:, :self.k-1]
        s = torch.sum(eta, dim=1)
        sum_backward = []
        for x in eta[:, :-1].T:
            s = s - x 
            sum_backward.append(s.view(-1))
        self.var_gamma2 = self.prior_alpha2 + torch.stack(sum_backward).T.float()
    

    def update_prior(self):

        self.prior_alpha1 = self.var_gamma1
        self.prior_alpha2 = self.var_gamma2
    
    def set_gamma(self, gamma1, gamma2):
        
        self.var_gamma1 = gamma1
        self.var_gamma2 = gamma2