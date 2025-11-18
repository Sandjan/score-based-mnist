import torch

def calc_loss(score_network: torch.nn.Module, x: torch.Tensor,sampler, use_sampler,y_cond,gen=None) -> tuple:
    # x: (batch_size, nch) is the training data
    
    # sample the time
    if use_sampler:
        # adaptive and based on loss distribution over time steps
        t = sampler.sample(x.shape[0], x.device).unsqueeze(-1)
    else:
        #uniform
        t = torch.rand((x.shape[0], 1), dtype=x.dtype, device=x.device,generator=gen) * (1 - 1e-4) + 1e-4

    # calculate the terms for the posterior log distribution
    int_beta = (0.1 + 0.5 * (20 - 0.1) * t) * t  # integral of beta
    mu_t = x * torch.exp(-0.5 * int_beta)
    var_t = -torch.expm1(-int_beta)
    x_t = torch.randn_like(x) * var_t ** 0.5 + mu_t
    grad_log_p = -(x_t - mu_t) / var_t  # (batch_size, nch)

    # calculate the score function
    score = score_network(x_t,y_cond, t)  # score: (batch_size, nch)

    # calculate the loss function
    loss = (score - grad_log_p) ** 2
    lmbda_t = var_t
    weighted_loss = lmbda_t * loss  # (batch_size, nch)
    
    loss_per_sample = torch.mean(weighted_loss, dim=1)  # (batch_size,)
    
    if sampler: #update timestep sampler with loss values
        sampler.update(t, loss_per_sample.detach())
    
    return torch.mean(loss_per_sample)