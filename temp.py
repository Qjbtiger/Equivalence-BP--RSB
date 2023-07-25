import numpy as np
import torch

def test(alpha, N, type):
        P = int(N*alpha)
        xi = np.random.choice([-1, 1], size=(P, N), replace=True, p=[0.5, 0.5]) # shape = (P, N)
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        torch.set_default_dtype(type)
        
        # define constant number & function
        eps = 1e-5
        maxIteration = 2000
        sqrtN = torch.sqrt(torch.tensor(N, device=device))
        N = torch.tensor(N, device=device)
        sqrt2 = torch.sqrt(torch.tensor(2, device=device))
        H = lambda x: torch.erfc(x/sqrt2) / 2

        # initialize
        m_i2nu = torch.rand(size=(N, P), device=device) * 2 - 1
        u_mu2i = torch.rand(size=(P, N), device=device) * 2 - 1
        w_mu2i = torch.zeros(size=(P, N), device=device)
        sigma_mu2i = torch.zeros(size=(P, N), device=device)
        m_i = torch.zeros(N, device=device)
        xi = torch.from_numpy(xi).to(device)

        for i in range(maxIteration):
            m_i2nu = torch.tanh(u_mu2i.sum(dim=0).reshape(-1, 1) - u_mu2i.T)
            w_mu2i = (torch.sum(m_i2nu.T*xi, dim=1)[:, None].reshape(-1, 1) - m_i2nu.T*xi) / sqrtN
            sigma_mu2i = (torch.sum(1-m_i2nu*m_i2nu, dim=0)[:, None].reshape(-1, 1) - torch.t(1-m_i2nu*m_i2nu)) / N
            u_mu2i = (torch.log(H(-(xi/sqrtN + w_mu2i) / (torch.sqrt(sigma_mu2i) + 1e-16)) + 1e-16) - torch.log(H(-(-xi/sqrtN + w_mu2i) / (torch.sqrt(sigma_mu2i) + 1e-16)) + 1e-16)) / 2

            last_mi = m_i
            m_i = torch.tanh(torch.sum(u_mu2i, dim=0))
            delta_mi = torch.max(torch.abs(last_mi-m_i)).cpu()
            print('N: {}, alpha: {:.4f}, delta_mi: {}, iteration: {}'.format(N, alpha, delta_mi, i))
            
import time

start = time.time()
test(1.02, 5000, torch.float32)
print('Single Float: {}'.format(time.time()-start))
start = time.time()
test(1.02, 5000, torch.float64)
print('Double float: {}'.format(time.time()-start))