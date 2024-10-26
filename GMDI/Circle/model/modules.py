import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class UConcenNet(nn.Module):

    def __init__(self, opt):
        super(UConcenNet, self).__init__()
        nh = opt.num_hidden
        nin = opt.u_dim
        nout = opt.u_dim
        self.fc1 = nn.Linear(nin, nh)
        self.fc2 = nn.Linear(nh, nout)

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        u = F.relu(self.fc1(x.float()))
        u = self.fc2(u)

        if re:
            u = u.reshape(T, B, -1)

        return u


class Theta2UNet(nn.Module):
    """
    input: 2 dim Theta
    output: 4 dim Theta for u loss
    """

    def __init__(self, opt):
        super(Theta2UNet, self).__init__()
        nin = opt.theta_dim
        nout = opt.u_dim

        self.fc = nn.Linear(nin, nout)

    def forward(self, x):
        return self.fc(x.float())


class ThetaNet(nn.Module):
    """
    Input: MSD(u)
    output: mean/var of theta
    """

    def __init__(self, opt):
        super(ThetaNet, self).__init__()
        nh = opt.num_hidden
        nin = opt.theta_dim
        self.opt = opt
        nout = nin
        self.nk = opt.k

        self.encoder = nn.Sequential(nn.Linear(nin, nh), nn.ReLU(inplace=True))

        self.fc_log_var = nn.Linear(nh, nout)

    def encode(self, x):
        result = self.encoder(x)
        log_var = self.fc_log_var(result)
        return log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        theta = mu + std * eps
        return theta

    def forward(self, x, mu):
        x = x.unsqueeze(dim=0).expand(self.nk, -1, -1) 
        mu = mu.unsqueeze(dim=0).expand(self.nk, -1, -1) 

        log_var = self.encode(x.float())
        theta = self.reparameterize(mu, log_var)
        return theta, log_var


class UNet(nn.Module):
    """
    Input: Data X
    Ouput: The estimated domain index
    Using Gaussian model
    """

    def __init__(self, opt):
        super(UNet, self).__init__()
        nh = opt.num_hidden
        nin = opt.input_dim
        n_u = opt.u_dim
        self.opt = opt

        nout = n_u
        self.encoder = nn.Sequential(
            nn.Linear(nin, nh),
            nn.ReLU(inplace=True),
            nn.Linear(nh, nh),
            nn.ReLU(inplace=True),
            nn.Linear(nh, nh),
            nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(nh, nout)
        self.fc_log_var = nn.Linear(nh, nout)

    def encode(self, x):
        result = self.encoder(x)
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        u = mu + std * eps
        return u

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        # u step is not reshaped!!
        mu, log_var = self.encode(x)
        u = self.reparameterize(mu, log_var)

        if re:
            u = u.reshape(T, B, -1)
            mu = mu.reshape(T, B, -1)
            log_var = log_var.reshape(T, B, -1)

        return u, mu, log_var


class DiscNet(nn.Module):
    """
    Discriminator doing binary classification: source v.s. target
    """

    def __init__(self, opt):
        super(DiscNet, self).__init__()
        nh = opt.num_hidden
        nout = opt.u_dim

        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()

        self.fc_final = nn.Linear(nh, nout)

    def forward(self, x):
        re = x.dim() == 4

        if re:
            K, T, B, C = x.shape
            x = x.reshape(K, T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = self.fc_final(x)

        if re:
            return x.reshape(K, T, B, -1)
        else:
            return x


class ClassDiscNet(nn.Module):
    """
    Discriminator doing multi-class classification on the domain
    """

    def __init__(self, opt):
        super(ClassDiscNet, self).__init__()
        nh = opt.num_hidden
        nin = nh
        nout = opt.num_domain

        print(f'===> Discriminator will distinguish {nout} domains')

        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()

        self.fc_final = nn.Linear(nh, nout)

    def forward(self, x):
        re = x.dim() == 4

        if re:
            K, T, B, C = x.shape
            x = x.reshape(K, T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))
        x = self.fc_final(x)
        x = torch.log_softmax(x, dim=2)
        if re:
            return x.reshape(K, T, B, -1)
        else:
            return x


class ReconstructNet(nn.Module):

    def __init__(self, opt):
        super(ReconstructNet, self).__init__()

        nh = opt.num_hidden
        nu = opt.u_dim
        nx = opt.input_dim  # the dimension of x

        self.fc1 = nn.Linear(nu, int(nh))
        self.fc2 = nn.Linear(int(nh), int(nh))
        self.fc3 = nn.Linear(int(nh), int(nh))
        self.fc_final = nn.Linear(int(nh), nx)

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc_final(x)

        if re:
            x = x.reshape(T, B, -1)
        return x


class Q_ZNet_theta(nn.Module):

    def __init__(self, opt):
        super(Q_ZNet_theta, self).__init__()

        nh = opt.num_hidden
        nu = opt.u_dim
        nx = opt.input_dim  # the dimension of x
        n_theta = opt.theta_dim
        self.opt = opt
        self.nk = opt.k

        self.fc1 = nn.Linear(nx, nh)
        self.fc2 = nn.Linear(nh * 3, nh * 2)
        self.fc3 = nn.Linear(nh * 2, nh * 2)
        self.fc_final = nn.Linear(nh * 2, nh)

        self.fc1_u = nn.Linear(nu, nh)
        self.fc2_u = nn.Linear(nh, nh)

        self.fc1_theta = nn.Linear(n_theta, nh)
        self.fc2_theta = nn.Linear(nh, nh)

        self.fc_q_mu = nn.Linear(nh, nh)
        self.fc_q_log_var = nn.Linear(nh, nh)

        self.fc_q_mu_2 = nn.Linear(nh, nh)
        self.fc_q_log_var_2 = nn.Linear(nh, nh)

        self.fc_p_mu = nn.Linear(nh, nh)
        self.fc_p_log_var = nn.Linear(nh, nh)

        self.fc_p_mu_2 = nn.Linear(nh, nh)
        self.fc_p_log_var_2 = nn.Linear(nh, nh)

    def encode(self, x, u, theta):
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(dim=0).expand(self.nk, -1, -1) 
        u = F.relu(self.fc1_u(u.float()))
        u = F.relu(self.fc2_u(u))
        u = u.unsqueeze(dim=0).expand(self.nk, -1, -1) 


        theta = F.relu(self.fc1_theta(theta.float()))
        theta = F.relu(self.fc2_theta(theta))


        tmp_B = int(u.shape[1] / theta.shape[1])

        theta = theta.unsqueeze(dim=2).expand(self.nk, -1, tmp_B,
                                            -1).reshape(self.nk, u.shape[1], -1)


        # combine feature in the middle
        x = torch.cat((x, u, theta), dim=2)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc_final(x))

        q_mu = F.relu(self.fc_q_mu(x))
        q_mu = self.fc_q_mu_2(q_mu)
        q_log_var = F.relu(self.fc_q_log_var(x))
        q_log_var = self.fc_q_log_var_2(q_log_var)

        p_mu = F.relu(self.fc_p_mu(x))
        p_mu = self.fc_p_mu_2(p_mu)
        p_log_var = F.relu(self.fc_p_log_var(x))
        p_log_var = self.fc_p_log_var_2(p_log_var)

        return q_mu, q_log_var, p_mu, p_log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        x = mu + std * eps
        return x

    def forward(self, x, u, theta):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            u = u.reshape(T * B, -1)

        q_mu, q_log_var, p_mu, p_log_var = self.encode(x, u, theta)
        q_z = self.reparameterize(q_mu, q_log_var)
        p_z = self.reparameterize(p_mu, p_log_var)

        if re:
            q_z = q_z.reshape(self.nk, T, B, -1)
            p_z = p_z.reshape(self.nk, T, B, -1)
            q_mu = q_mu.reshape(self.nk, T, B, -1)
            q_log_var = q_log_var.reshape(self.nk, T, B, -1)
            p_mu = p_mu.reshape(self.nk, T, B, -1)
            p_log_var = p_log_var.reshape(self.nk, T, B, -1)
        return q_z, q_mu, q_log_var, p_z, p_mu, p_log_var


class Q_ZNet(nn.Module):

    def __init__(self, opt):
        super(Q_ZNet, self).__init__()

        nh = opt.num_hidden
        nu = opt.u_dim
        nx = opt.input_dim  # the dimension of x
        self.opt = opt

        self.fc1 = nn.Linear(nx, nh)
        self.fc2 = nn.Linear(nh * 2, nh * 2)
        self.fc3 = nn.Linear(nh * 2, nh * 2)
        self.fc_final = nn.Linear(nh * 2, nh)

        self.fc1_u = nn.Linear(nu, nh)
        self.fc2_u = nn.Linear(nh, nh)

        self.fc_q_mu = nn.Linear(nh, nh)
        self.fc_q_log_var = nn.Linear(nh, nh)

        self.fc_q_mu_2 = nn.Linear(nh, nh)
        self.fc_q_log_var_2 = nn.Linear(nh, nh)

        self.fc_p_mu = nn.Linear(nh, nh)
        self.fc_p_log_var = nn.Linear(nh, nh)

        self.fc_p_mu_2 = nn.Linear(nh, nh)
        self.fc_p_log_var_2 = nn.Linear(nh, nh)

    def encode(self, x, u):
        x = F.relu(self.fc1(x))
        u = F.relu(self.fc1_u(u.float()))
        u = F.relu(self.fc2_u(u))

        # combine feature in the middle
        x = torch.cat((x, u), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc_final(x))

        q_mu = F.relu(self.fc_q_mu(x))
        q_mu = self.fc_q_mu_2(q_mu)
        q_log_var = F.relu(self.fc_q_log_var(x))
        q_log_var = self.fc_q_log_var_2(q_log_var)

        p_mu = F.relu(self.fc_p_mu(x))
        p_mu = self.fc_p_mu_2(p_mu)
        p_log_var = F.relu(self.fc_p_log_var(x))
        p_log_var = self.fc_p_log_var_2(p_log_var)

        return q_mu, q_log_var, p_mu, p_log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        x = mu + std * eps
        return x

    def forward(self, x, u):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)
            u = u.reshape(T * B, -1)

        q_mu, q_log_var, p_mu, p_log_var = self.encode(x, u)
        q_z = self.reparameterize(q_mu, q_log_var)
        p_z = self.reparameterize(p_mu, p_log_var)

        if re:
            q_z = q_z.reshape(T, B, -1)
            p_z = p_z.reshape(T, B, -1)
            q_mu = q_mu.reshape(T, B, -1)
            q_log_var = q_log_var.reshape(T, B, -1)
            p_mu = p_mu.reshape(T, B, -1)
            p_log_var = p_log_var.reshape(T, B, -1)
        return q_z, q_mu, q_log_var, p_z, p_mu, p_log_var


class PredNet(nn.Module):

    def __init__(self, opt):
        # This is for classification task.
        super(PredNet, self).__init__()
        nh, nc = opt.num_hidden, opt.num_class
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)
        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)
        self.fc_final = nn.Linear(nh, nc)
        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()

    def forward(self, x, return_softmax=False):
        re = x.dim() == 4
        if re:
            K, T, B, C = x.shape
            x = x.reshape(K, T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc_final(x)

        x_softmax = F.softmax(x, dim=2)
        x = torch.log(x_softmax + 1e-4)
        if re:
            x = x.reshape(K, T, B, -1)
            # x_softmax = x_softmax.reshape(K, T, B, -1)

        if return_softmax:
            return x, x_softmax
        else:
            return x


class GraphDNet(nn.Module):
    """
    Generate z' for connection loss
    """

    def __init__(self, opt):
        super(GraphDNet, self).__init__()
        nh = opt.num_hidden
        nin = nh
        self.fc3 = nn.Linear(nin, nh)
        self.bn3 = nn.BatchNorm1d(nh)

        self.fc4 = nn.Linear(nh, nh)
        self.bn4 = nn.BatchNorm1d(nh)

        self.fc5 = nn.Linear(nh, nh)
        self.bn5 = nn.BatchNorm1d(nh)

        self.fc6 = nn.Linear(nh, nh)
        self.bn6 = nn.BatchNorm1d(nh)

        self.fc7 = nn.Linear(nh, nh)
        self.bn7 = nn.BatchNorm1d(nh)

        self.fc_final = nn.Linear(nh, opt.u_dim)

        if opt.no_bn:
            self.bn3 = Identity()
            self.bn4 = Identity()
            self.bn5 = Identity()
            self.bn6 = Identity()
            self.bn7 = Identity()

    def forward(self, x):
        re = x.dim() == 3

        if re:
            T, B, C = x.shape
            x = x.reshape(T * B, -1)

        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        x = F.relu(self.bn7(self.fc7(x)))

        x = self.fc_final(x)

        if re:
            return x.reshape(T, B, -1)
        else:
            return x
