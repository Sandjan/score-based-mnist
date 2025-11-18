import torch
import torch.nn.functional as F
import torchvision
import time
from dataclasses import dataclass
import matplotlib.pyplot as plt

from src.network import ScoreNetwork0
from src.tools import AdaptiveTimeSampler,generate_grid_samples
from src.loss import calc_loss


@dataclass
class Config:
    min_batch_size: int = 20
    max_batch_size: int = 74
    initial_beta1: float = 0.8687328416419
    final_beta1: float = 0.712165096747
    init_lr: float = 0.002889579643
    min_lr: float = 7.5730446e-05
    weight_decay: float = 2.23943797e-05
    total_train_steps: int = 77000
    log_step_size: int = 1000
    batch_scheduler_alpha: float = 0.8846055990073
    time_sampler_bins: int = 20
    time_sampler_sigma: float = 0.2042285013114
    time_sampler_alpha: float = 0.1045238579512
    time_sampler_start: int = 6292
    ema_model_alpha: float = 0.0393098317891
    clip_grad_norm: float = 1.1120015678523

def evaluate_on_test(model,x_test,y_test,batch_size=64):
    model.eval()
    loss_total = 0
    steps = 0
    for i in range(0,x_test.size(0)-batch_size,batch_size):
        batch = (x_test[i:i+batch_size].float()/255.0)
        y_cond = y_test[i:i+batch_size]
        batch = batch.reshape(batch.shape[0], -1)
        loss_total += calc_loss(model, batch,None,False,y_cond).item()
        steps+=1
    return loss_total/steps


torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
cfg = Config()

def get_batch_size(step):
    # Schedules the batch_size for faster but noisier steps at the beginning and slower but more stable steps at the end.
    t = (step/cfg.total_train_steps)
    linear_part = cfg.min_batch_size + t * (cfg.max_batch_size - cfg.min_batch_size)
    exp_part = cfg.min_batch_size * (cfg.max_batch_size/cfg.min_batch_size)**t
    return int((1-cfg.batch_scheduler_alpha) * linear_part + cfg.batch_scheduler_alpha * exp_part)


mnist_dset = torchvision.datasets.MNIST("mnist", download=True)
mnist_test = torchvision.datasets.MNIST("mnist",train=False)
score_network = ScoreNetwork0()

# time sampler for adaptive sampling of time steps
t_sampler = AdaptiveTimeSampler(cfg.time_sampler_bins,cfg.time_sampler_sigma,cfg.time_sampler_alpha,device='cuda')

# MNIST is small -> transfer everything on the gpu for fast training
x_gpu_u8 = mnist_dset.data.unsqueeze(1).contiguous().to(dtype=torch.uint8).cuda()  # [N,1,28,28]
N = x_gpu_u8.size(0)
y_onehot = F.one_hot(mnist_dset.targets, num_classes=10).to(torch.bool).cuda()

x_test_u8 = mnist_test.data.unsqueeze(1).contiguous().to(dtype=torch.uint8).cuda()
y_test_onehot = F.one_hot(mnist_test.targets, num_classes=10).to(torch.bool).cuda()

# start the training loop
opt = torch.optim.AdamW(score_network.parameters(), lr=cfg.init_lr,weight_decay=cfg.weight_decay, betas=(cfg.initial_beta1, 0.999))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,cfg.total_train_steps,cfg.min_lr)
score_network = score_network.cuda()

# Exponential Moving Average for Stochastic Weight Averaging for better results
ema_params = {k: v.clone().detach() for k, v in score_network.state_dict().items()} 
test_y = F.one_hot(torch.arange(10, device="cuda"),num_classes=10)
total_loss = 0
total_time = 0
for step in range(cfg.total_train_steps):
    start = time.time()
    opt.zero_grad(set_to_none=True)
    idx = torch.randint(N, (get_batch_size(step),),device="cuda")
    batch = (x_gpu_u8[idx].float()/255.0)
    y_cond = y_onehot[idx]
    batch = batch.reshape(batch.shape[0], -1) #weird reshaping going on
    # training step
    loss = calc_loss(score_network, batch,t_sampler,step>cfg.time_sampler_start,y_cond)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(score_network.parameters(), cfg.clip_grad_norm)
    opt.step()
    scheduler.step()

    # running stats
    total_loss += loss.detach().item()

    total_time += time.time()-start

    # print the training stats
    if step % cfg.log_step_size == 0:
        print("Step:", step,"/",cfg.total_train_steps)
        print("Train loss: ",total_loss / cfg.log_step_size)
        total_loss = 0
        # Schedule beta1 of the optimizer to achieve greater stability with small batch sizes and lower inertia toward the optimum
        beta1 = cfg.initial_beta1 - (cfg.initial_beta1 - cfg.final_beta1) * step / cfg.total_train_steps
        for g in opt.param_groups:
            g['betas'] = (beta1, g['betas'][1])

    if step%100==0:
        with torch.no_grad():
            for k, v in score_network.state_dict().items():
                ema_params[k].mul_(cfg.ema_model_alpha).add_(v.detach() * (1 - cfg.ema_model_alpha))

    if step%(cfg.log_step_size*10) == 0 and not step==0:
        print("Test loss: ",evaluate_on_test(score_network,x_test_u8,y_test_onehot))
        
    if step%(cfg.log_step_size*20) == 0 and not step==0:
        print("Generating test image at step:",step)
        img = generate_grid_samples(score_network,test_y) #generate example images from 0-9
        plt.imsave(f"./images/samples_step{step}.png", img,cmap="gray")

score_network.load_state_dict(ema_params)
print("Test loss after SWA: ",evaluate_on_test(score_network,x_test_u8,y_test_onehot))
torch.save(ema_params, "ema_params.pt")
img = generate_grid_samples(score_network,test_y)
plt.imsave(f"./images/samples_final.png", img,cmap="gray")