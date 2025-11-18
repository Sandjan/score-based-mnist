import torch
from torchvision.utils import make_grid

def generate_samples(score_network: torch.nn.Module, nsamples: int,y_cond) -> torch.Tensor:
    device = next(score_network.parameters()).device
    x_t = torch.randn((nsamples, 28 * 28), device=device)  # (nsamples, nch)
    time_pts = torch.linspace(1, 0, 1000, device=device)  # (ntime_pts,)
    beta = lambda t: 0.1 + (20 - 0.1) * t
    for i in range(len(time_pts) - 1):
        t = time_pts[i]
        dt = time_pts[i + 1] - t

        # calculate the drift and diffusion terms
        fxt = -0.5 * beta(t) * x_t
        gt = beta(t) ** 0.5
        score = score_network(x_t,y_cond, t.expand(x_t.shape[0], 1)).detach()
        drift = fxt - gt * gt * score
        diffusion = gt

        # euler-maruyama step
        x_t = x_t + drift * dt + diffusion * torch.randn_like(x_t) * torch.abs(dt) ** 0.5
    return x_t

def generate_grid_samples(model,test_y):
    samples = generate_samples(model, 10,test_y).detach().reshape(-1, 28, 28).unsqueeze(1)
    grid = make_grid(samples, nrow=samples.shape[0], padding=2, normalize=True, value_range=(0,1))
    grid_img = grid.mul(255).byte().cpu()

    return grid_img[0].numpy()

class AdaptiveTimeSampler:
    """
    Adaptive time sampler for diffusion models.
    
    Tracks the loss distribution across bins and samples t-values
    preferentially from regions with higher loss.
    
    The sampler maintains a learned distribution of loss values across the
    time domain [t_min, t_max]. It uses a kernel-based update mechanism to
    smooth observed losses into bins, then samples new time values with
    probability proportional to the estimated loss in each bin.
    
    Args:
        num_bins (int): Number of bins to divide the time domain into.
        sigma (float): Kernel bandwidth for smoothing losses into bins.
            Smaller values = more localized updates, larger values = more smoothing.
        alpha (float): Exponential moving average factor for loss updates.
            Higher values = faster adaptation to new observations.
        t_min (float): Minimum time value (typically 0.0).
        t_max (float): Maximum time value (typically 1.0).
        device (str): Torch device to use ('cpu' or 'cuda').
        dtype (torch.dtype): Data type for tensors (e.g., torch.float32).
    """
    
    def __init__(self, num_bins=20, sigma=0.01, alpha=0.05, 
                 t_min=0.0, t_max=1.0, device='cpu', dtype=torch.float32):
        self.num_bins = num_bins
        self.sigma = sigma
        self.alpha = alpha
        self.t_min = t_min
        self.t_max = t_max
        self.device = device
        self.dtype = dtype
        
        # Bin edges and centers
        # Create edges first: [t_min, ..., t_max] with num_bins+1 points
        self.edges = torch.linspace(t_min, t_max, num_bins + 1, 
                                   device=device, dtype=dtype)
        self.bin_width = (self.edges[1] - self.edges[0]).item()
        
        # Calculate bin centers as midpoints between edges
        # This ensures each center is in the middle of its bin
        self.bin_centers = (self.edges[:-1] + self.edges[1:]) / 2
        
        # Loss distribution (initialized uniformly)
        self.loss_dist = torch.ones(num_bins, device=device, dtype=dtype)
        
        # Sampling probabilities (updated with each update)
        self.sampling_probs = torch.ones(num_bins, device=device, dtype=dtype) / num_bins
        
    def update(self, t_values, losses):
        """
        Update the loss distribution based on observed (t, loss) pairs.
        
        Uses a Gaussian kernel to smooth each observed loss into nearby bins,
        then updates the loss distribution using exponential moving average.
        This allows the sampler to build a continuous estimate of where
        losses are highest in the time domain.
        
        Args:
            t_values (torch.Tensor): Shape (N,) with t-values where losses were observed.
            losses (torch.Tensor): Shape (N,) with corresponding loss values.
        """

        t_values = t_values.flatten()
        losses = losses.flatten()
        # Calculate kernel weights
        # Distance from each observed t to each bin center
        dists = self.bin_centers.unsqueeze(0) - t_values.unsqueeze(1)  # (N, bins)
        
        # Gaussian kernel: high weight for nearby bins, low for distant bins
        K = torch.exp(-0.5 * (dists / self.sigma)**2)  # (N, bins)
        
        # Ensure losses are non-negative
        weights = losses.clamp_min(0.0)  # (N,)
        
        # Weighted average per bin
        # Each bin accumulates weighted losses from all observations
        numerator = (K * weights.unsqueeze(1)).sum(dim=0)  # (bins,)
        denominator = K.sum(dim=0).clamp_min(1e-12)  # (bins,)
        delta = numerator / denominator  # (bins,)
        
        # Exponential moving average
        # Smoothly incorporate new observations while retaining history
        self.loss_dist = (1 - self.alpha) * self.loss_dist + self.alpha * delta
        
        # Update sampling probabilities (higher loss = higher probability)
        # Optional: Add temperature parameter for stronger/weaker focusing
        probs = self.loss_dist.clamp_min(1e-12)
        self.sampling_probs = probs / probs.sum()
        
    def sample(self, batch_size, device=None):
        """
        Sample t-values based on the current loss distribution.
        
        Regions with higher estimated loss are sampled more frequently.
        This focuses training compute on the most difficult parts of the
        time domain.
        
        Args:
            batch_size (int): Number of t-values to sample.
            device (str, optional): Overrides self.device if provided.
            
        Returns:
            torch.Tensor: Shape (batch_size,) with sampled t-values in [t_min, t_max].
        """
        if device is None:
            device = self.device
            
        # Sample bins based on loss distribution
        # Bins with higher loss are chosen more frequently
        bin_indices = torch.multinomial(
            self.sampling_probs, 
            batch_size, 
            replacement=True
        )
        
        # Uniform sampling within the chosen bin
        bin_starts = self.edges[bin_indices]
        bin_ends = self.edges[bin_indices + 1]
        
        # Uniform distribution within [bin_start, bin_end)
        uniform_offset = torch.rand(batch_size, device=device, dtype=self.dtype)
        t_samples = bin_starts + uniform_offset * (bin_ends - bin_starts)
        
        return t_samples * (1 - 1e-4) + 1e-4
    
    def get_loss_dist(self):
        """
        Returns the current loss distribution.
        
        Returns:
            torch.Tensor: Shape (num_bins,) with estimated loss per bin.
        """
        return self.loss_dist.clone()