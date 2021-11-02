# Phase harmonic sythesis script using pywph package

import numpy as np
import time
import torch
import scipy.optimize as opt
import pywph as pw
import multiprocessing
print('CPUs"',multiprocessing.cpu_count())
print('GPU count: ' + str(torch.cuda.device_count()) + '\n')

class SoftHistogram(torch.nn.Module):
    """
    Motivated by https://discuss.pytorch.org/t/differentiable-torch-histc/25865/3
    """
    def __init__(self, bins, min_bin_edge, max_bin_edge, sigma):
        super(SoftHistogram, self).__init__()
        self.sigma = sigma
        self.delta = float(max_bin_edge - min_bin_edge) / float(bins)
        self.centers = float(min_bin_edge) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = torch.nn.Parameter(self.centers, requires_grad=False).to(device)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta/2)) - torch.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        return x

#######
# INPUT PARAMETERS
#######

M, N = 512, 512
J = 8
L = 8 
dn = 2
dj=None # default J - 1
dl=None # default L/2

norm = "auto"   # Normalization
pbc = False      # Periodic boundary conditions

device = 1
print(torch.cuda.device(device))

optim_params = {"maxiter": 2000, "gtol": 1e-20, "ftol": 1e-20, "maxcor": 20}

p_list = [0, 1, 2, 3, 4, 5, 6]

data = np.load('./data/sim_QiU_stdr_3.npy').astype(np.complex64)
output_filename = './data/demo_synth_'

#######
# PREPARATION AND INITIAL GUESS
#######

# Normalize input data
data_std_real = data.real.std()
data_std_imag = data.imag.std()
data_mean_real = data.real.mean()
data_mean_imag = data.imag.mean()
data  = (data.real - data_mean_real)/data_std_real + 1j* (data.imag - data_mean_imag)/data_std_imag

if pbc == True:
    data_torch = torch.from_numpy(data)
else:
    data_torch = torch.from_numpy(data[int(M/4):int(-M/4),int(N/4):int(-N/4)])

data_torch = data_torch.to(device)


softhist = SoftHistogram(bins=128, min_bin_edge=-5, max_bin_edge=5, sigma=20)

target_hist_real = softhist(data_torch.real.flatten())
target_hist_imag = softhist(data_torch.imag.flatten())

target_hist_real = target_hist_real.to(device)
target_hist_imag = target_hist_imag.to(device)

target_hist_real = target_hist_real/torch.sum(target_hist_real)
target_hist_imag = target_hist_imag/torch.sum(target_hist_imag)
    

print(target_hist_real, torch.sum(target_hist_real))
print(target_hist_imag, torch.sum(target_hist_imag))

cplx = np.iscomplexobj(data)

print("Building operator...", flush=True)
start_time = time.time()
wph_op = pw.WPHOp(M, N, J, L=L, dn=dn, device=device)
print("Load model...", flush=True)
wph_op.load_model(p_list=p_list,dl=dl,dj=dj)
print(f"Done! (in {time.time() - start_time}s)", flush=True)

print("Computing stats of target image...", flush=True)
start_time = time.time()
if cplx:
    coeffs = wph_op.apply(np.stack((data, data.real+1j*1e-9, data.imag+1j*1e-9), axis=0), norm=norm, padding=not pbc)
else:
    coeffs = wph_op.apply(data, norm=norm, padding=not pbc)
print(f"Done! (in {time.time() - start_time}s)")

#######
# SYNTHESIS
#######

eval_cnt = 0


def objective(x):
    global eval_cnt
    print(f"Evaluation: {eval_cnt}")
    start_time = time.time()
    
    # Reshape x
    if cplx:
        x_curr = x.reshape((M, N, 2)).astype(np.float32)
        x_curr = x_curr[..., 0] + 1j*x_curr[..., 1]
    else:
        x_curr = x.reshape((M, N)).astype(np.float32)
    x_curr = torch.from_numpy(x_curr).to(device)
    x_curr.requires_grad = True
    
    if cplx:
        x_curr_wreal = torch.stack((x_curr, x_curr.real+1j*1e-9, x_curr.imag+1j*1e-9), axis=0)
    else:
        x_curr_wreal = x_curr
    
    # Compute the loss (squared 2-norm)
    loss_tot = torch.zeros(1)
#     x_curr_wreal, nb_chunks = wph_op.preconfigure(x_curr_wreal, mem_chunk_factor_grad=80)
    x_curr_wreal, nb_chunks = wph_op.preconfigure(x_curr_wreal, precompute_wt=False, precompute_modwt=False, mem_chunk_factor_grad=20)
    for i in range(nb_chunks):
        coeffs_chunk, indices = wph_op.apply(x_curr_wreal, i, norm=norm, ret_indices=True)
        loss = torch.sum(torch.abs(coeffs_chunk - coeffs[..., indices])**2. )
        loss.backward(retain_graph=True)
        loss_tot += loss.detach().cpu()
        del coeffs_chunk, indices, loss
        
    # Compute histogram loss
    lambda_hist = 5e4 # lagrange multiplier
    hist_real = softhist(x_curr.real.flatten())
    hist_imag = softhist(x_curr.imag.flatten())

    hist_real = hist_real/torch.sum(hist_real)
    hist_imag = hist_imag/torch.sum(hist_imag)

    epsilon_log = 1e-8
    loss = torch.sum(target_hist_real*torch.log(target_hist_real) - target_hist_real*torch.log(hist_real + epsilon_log))
    loss += torch.sum(target_hist_imag*torch.log(target_hist_imag) - target_hist_imag*torch.log(hist_imag + epsilon_log))
    loss = loss*lambda_hist
    loss.backward(retain_graph=True)
    loss_tot +=  loss.detach().cpu()
    
    # Reshape the gradient
    if cplx:
        x_grad = np.zeros_like(x).reshape((M, N, 2))
        x_grad[:, :, 0] = x_curr.grad.real.cpu().numpy()
        x_grad[:, :, 1] = x_curr.grad.imag.cpu().numpy()
    else:
        x_grad = x_curr.grad.cpu().numpy().astype(x.dtype)
    
    print(f"Histogram loss: {loss.item()}")
    print(f"Total loss: {loss_tot.item()} (computed in {time.time() - start_time}s)")
    print('sum x_grad real: ' + str(np.sum(x_grad[:, :, 0].flatten())))
    print('sum x_grad imag: ' + str(np.sum(x_grad[:, :, 1].flatten())), flush=True)
    eval_cnt += 1
    return loss_tot.item(), x_grad.ravel()


for n_iter in np.arange(270,280):
    total_start_time = time.time()

    # Initial guess
    if cplx:
        x0 = np.zeros((M, N, 2), dtype=np.float64)
        x0[:, :, 0] = np.random.normal(data.real.mean(), data.real.std(), data.shape)
        x0[:, :, 1] = np.random.normal(data.imag.mean(), data.imag.std(), data.shape)
    else:
        x0 = np.random.normal(data.mean(), data.std(), data.shape)

    result = opt.minimize(objective, x0.ravel(), method='L-BFGS-B', jac=True, tol=None, options=optim_params)
    final_loss, x_final, niter, msg = result['fun'], result['x'], result['nit'], result['message']

    print(f"Synthesis ended in {niter} iterations with optimizer message: {msg}")
    print(f"Synthesis time: {time.time() - total_start_time}s")

    #######
    # OUTPUT
    #######

    if cplx:
        x_final = x_final.reshape((M, N, 2)).astype(np.float32)
        x_final = x_final[..., 0] + 1j*x_final[..., 1]
    else:
        x_final = x_final.reshape((M, N)).astype(np.float32)

    x_final = x_final.real*data_std_real + data_mean_real + 1j*(x_final.imag*data_std_imag + data_mean_imag)

    if output_filename is not None:
        np.save(output_filename + str(n_iter) + 'npy', x_final)