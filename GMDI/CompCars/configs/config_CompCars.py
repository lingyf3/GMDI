from easydict import EasyDict
from time import localtime, strftime
# set experiment configs
opt = EasyDict()

opt.data_src = "data/"
opt.data_path = opt.data_src + "feature_resnet18_new_data.pkl"

opt.src_domain_idx = [0]
opt.src_domain_id = 0
opt.tgt_domain_idx = list(set(range(0, 30)) - set(opt.src_domain_idx))

opt.num_source = len(opt.src_domain_idx)
opt.num_target = len(opt.tgt_domain_idx)
opt.num_domain = opt.num_source + opt.num_target

opt.all_domain_idx = opt.src_domain_idx + opt.tgt_domain_idx

# wheather shuffle data
opt.shuffle = True

opt.use_pretrain_R = False
# opt.pretrain_R_path = "data/netR_8_434.pth"
# opt.pretrain_U_path = "data/netU_8_434.pth"
opt.fix_u_r = False
opt.use_pretrain_model_all = False

opt.d_loss_type = "ADDA_loss"  # "DANN_loss_mean" # "CIDA_loss" # "GRDA_loss" # "DANN_loss"

opt.lambda_gan = 0.6
opt.lambda_reconstruct = 1000
opt.lambda_u_concentrate = 0.8
opt.lambda_u_theta = 0.8
opt.lambda_theta = 0.8

# for warm up
opt.init_lr = 1e-6
opt.peak_lr_e = 2.9 * 1e-4
opt.peak_lr_d = 2.9 * 1e-4
opt.final_lr = 1e-8
opt.warmup_steps = 60

opt.seed = 2333
opt.num_epoch = 500
opt.batch_size = 16

opt.use_visdom = False  # True
opt.visdom_port = 2000
tmp_time = localtime()
opt.outf = "result_save/{}".format(strftime("%Y-%m-%d %H:%M:%S", tmp_time))
opt.save_interval = 500
opt.test_interval = 20

opt.device = "cuda:1"
opt.gpu_device = "0, 1, 2, 3, 4, 5, 6, 7"
opt.gamma = 100
opt.theta1 = 0.9
opt.weight_decay = 5e-4
opt.normalize_domain = False
opt.no_bn = True  # do not use batch normalization

# network parameter
opt.num_hidden = 2048
opt.num_class = 4
opt.input_dim = 512  # the dimension of input data x

opt.u_dim = 8  # the dimension of local domain index u
opt.theta_dim = 2
opt.k = 3
opt.alpha = 1 #  concentration parameter

# for grda discriminator
opt.sample_v = 20

# how many nodes to save
opt.save_sample = 100
