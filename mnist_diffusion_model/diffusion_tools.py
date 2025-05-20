import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from tqdm import tqdm
import os
from .models import ContextUnet
from .utils import generate_animation

####train
def set_bn_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
def perturb_input(x, t, noise, ab_t):
    return ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]).sqrt() * noise
def get_masked_context(context, p=0.9):
    return context*torch.bernoulli(torch.ones((context.shape[0], 1))*p)

def generator_train(nn_model, data_iter, g_local_steps, lr=1e-3, timesteps=500, beta1=1e-4, beta2=0.02):
        nn_model.cuda()
        nn_model.train()  
        nn_model.apply(set_bn_eval)
        _ , _, ab_t = get_ddpm_noise_schedule(timesteps, beta1, beta2)
        #dataloader = DataLoader(dataset, batch_size, True)   
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, nn_model.parameters()), lr=lr)
        #scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1, end_factor=0.01, total_iters=50)
        for _ in range(g_local_steps):
            x, c, _ = data_iter.next()
            x = x.cuda()
            c = get_masked_context(c).cuda()
            noise = torch.randn_like(x)
            t = torch.randint(1, timesteps + 1, (x.shape[0], )).cuda()
            x_pert = perturb_input(x, t, noise, ab_t)
            pred_noise = nn_model(x_pert, t / timesteps, c=c)
            loss = torch.nn.functional.mse_loss(pred_noise, noise)
            optim.zero_grad()
            loss.backward()
            optim.step()
        #scheduler.step()
        return {name: param for name, param in nn_model.state_dict().items() if 'contextembs' in name}

#####sample
def get_custom_context(n_samples, n_classes, target_class=None):
    if target_class is not None:
        assert 0 <= target_class < n_classes
        context = [target_class] * n_samples
    else:
        context = []
        for i in range(n_classes - 1):
            context.extend([i]*(n_samples//n_classes))
        context.extend([n_classes - 1]*(n_samples - len(context)))
    return torch.nn.functional.one_hot(torch.tensor(context), n_classes).float().cuda()

def MNIST_get_learned_conditioning(n_samples, cls_id):
    context = []
    context.extend([cls_id]*n_samples)
     
def get_ddpm_noise_schedule(timesteps, beta1, beta2):
    b_t = torch.linspace(beta1, beta2, timesteps+1, device='cuda')
    a_t = 1 - b_t
    ab_t = torch.cumprod(a_t, dim=0)
    ab_t[0] = 1
    return a_t, b_t, ab_t
def denoise_add_noise(x, t, pred_noise, a_t, b_t, ab_t, z):
    noise = b_t.sqrt()[t]*z
    mean = (x - pred_noise * ((1 - a_t[t]) / (1 - ab_t[t]).sqrt())) / a_t[t].sqrt()
    return mean + noise

def denoise_ddim(x, t, t_prev, pred_noise, a_t, b_t, ab_t):
    ab = ab_t[t]
    ab_prev = ab_t[t_prev]
    x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
    dir_xt = (1 - ab_prev).sqrt() * pred_noise
    return x0_pred + dir_xt


@torch.no_grad()
def MNIST_sample_ddim(nn_model, n_samples, context=None, timesteps=None, 
                    beta1=None, beta2=None, n=10):
    a_t, b_t, ab_t = get_ddpm_noise_schedule(timesteps, beta1, beta2)
    nn_model.eval()
    samples = torch.randn(n_samples, nn_model.in_channels, 
                        nn_model.height, nn_model.width, 
                        device='cuda')
    step_size = timesteps // n
    for t in range(timesteps, 0, -step_size):#timesteps .... 1
        eps  = nn_model(samples, torch.tensor([t/timesteps], device='cuda')[:, None, None, None], context)# predict noise e_(x_t,t,c)
        samples = denoise_ddim(samples, t, t-step_size, eps, a_t, b_t, ab_t)
    x_samples_ddpm = torch.clamp((samples.detach()+1.0)/2.0, min=0.0, max=1.0)
    return x_samples_ddpm
@torch.no_grad()
def MNIST_sample_ddpm(nn_model, n_samples, context=None, timesteps=None, 
                    beta1=None, beta2=None):
    a_t, b_t, ab_t = get_ddpm_noise_schedule(timesteps, beta1, beta2)
    nn_model.eval()
    samples = torch.randn(n_samples, nn_model.in_channels, 
                        nn_model.height, nn_model.width, 
                        device='cuda')
    for t in range(timesteps, 0, -1):#timesteps .... 1
        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if t > 1 else 0
        eps  = nn_model(samples, torch.tensor([t/timesteps], device='cuda')[:, None, None, None], context)# predict noise e_(x_t,t,c)
        samples = denoise_add_noise(samples, t, eps, a_t, b_t, ab_t, z)
    x_samples_ddpm = torch.clamp((samples.detach()+1.0)/2.0, min=0.0, max=1.0)
    return x_samples_ddpm


def sample_ddpm(nn_model, n_samples, context=None, timesteps=None, 
                    beta1=None, beta2=None, save_rate=20, inference_transform=lambda x: (x+1)/2):

        a_t, b_t, ab_t = get_ddpm_noise_schedule(timesteps, beta1, beta2)
        nn_model.eval()
        samples = torch.randn(n_samples, nn_model.in_channels, 
                              nn_model.height, nn_model.width, 
                              device='cuda')
        intermediate_samples = [samples.detach().cpu()] # samples at T = timesteps
        t_steps = [timesteps] # keep record of time to use in animation generation
        for t in range(timesteps, 0, -1):
            print(f"Sampling timestep {t}", end="\r")
            if t % 50 == 0: print(f"Sampling timestep {t}")

            z = torch.randn_like(samples) if t > 1 else 0
            pred_noise = nn_model(samples, torch.tensor([t/timesteps], device='cuda')[:, None, None, None], context)
            samples = denoise_add_noise(samples, t, pred_noise, a_t, b_t, ab_t, z)
            
            if t % save_rate == 1 or t < 8:
                intermediate_samples.append(inference_transform(samples.detach().cpu()))
                t_steps.append(t-1)
        return intermediate_samples[-1], intermediate_samples, t_steps



def generate(nn_model, n_samples, n_images_per_row, timesteps, beta1, beta2):
    root = "./mnist_generated_images"
    os.makedirs(root, exist_ok=True)
    with torch.no_grad():
        x0, intermediate_samples, t_steps = sample_ddpm(nn_model, n_samples, get_custom_context(
                                                            n_samples, nn_model.n_cfeat),
                                                            timesteps,beta1,beta2,)
    save_image(x0, os.path.join(root, "ddpm_images.jpeg"), nrow=n_images_per_row)
    generate_animation(intermediate_samples,
                        t_steps, 
                        os.path.join(root, f"animation.gif"),
                        n_images_per_row)