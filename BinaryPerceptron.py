import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from scipy import special, optimize, integrate
from QLog import qLog

class BinaryPerceptron(object):
    def __init__(self, N):
        self.N = N
        self.P = None
        self.xi = None

    def generateData(self, alpha):
        self.alpha = alpha
        self.P = int(self.N*alpha)
        self.xi = np.random.choice([-1, 1], size=(self.P, self.N), replace=True, p=[0.5, 0.5]) # shape = (P, N)

    def messagePassing(self):
        # check is data existed
        if not self.P:
            print('data is not existed!')
            exit(1)
        
        # define constant number & function
        eps = 1e-5
        maxIteration = 1000
        sqrtN = np.sqrt(self.N)
        sqrt2 = np.sqrt(2)
        H = lambda x: special.erfc(x/sqrt2) / 2

        # initialize
        m_i2nu = np.random.uniform(-1, 1, size=(self.N, self.P))
        u_mu2i = np.random.uniform(-1, 1, size=(self.P, self.N))
        w_mu2i = np.random.uniform(-1, 1, size=(self.P, self.N))
        sigma_mu2i = np.random.uniform(-1, 1, size=(self.P, self.N))
        m_i = np.random.uniform(-1, 1, size=self.N)

        flag = False
        for i in range(maxIteration):
            m_i2nu = np.tanh(u_mu2i.sum(axis=0)[:, np.newaxis].repeat(self.P, axis=1) - u_mu2i.T)
            w_mu2i = (np.sum(m_i2nu.T*self.xi, axis=1)[:, np.newaxis].repeat(self.N, axis=1) - m_i2nu.T*self.xi) / sqrtN
            sigma_mu2i = (np.sum(1-m_i2nu*m_i2nu, axis=0)[:, np.newaxis].repeat(self.N, axis=1) - (1-m_i2nu*m_i2nu).T) / self.N
            u_mu2i = (np.log(H(-(self.xi/sqrtN + w_mu2i) / (np.sqrt(sigma_mu2i) + 1e-10)) + 1e-10) - np.log(H(-(-self.xi/sqrtN + w_mu2i) / (np.sqrt(sigma_mu2i) + 1e-10)) + 1e-10)) / 2

            last_mi = m_i
            m_i = np.tanh(np.sum(u_mu2i, axis=0))
            delta_mi = np.max(np.fabs(last_mi-m_i))
            qLog.weakInfo('N: {}, alpha: {:.3f}, delta_mi: {}, iteration: {}'.format(self.N, self.alpha, delta_mi, i))

            if (delta_mi < eps):
                return True, i+1
        
        return flag, maxIteration

    def messagePassingGPU(self):
        # check is data existed
        if not self.P:
            qLog.info('Data is not existed!')
            exit(1)
        
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        # define constant number & function
        eps = 1e-5
        maxIteration = 2000
        sqrtN = torch.sqrt(torch.tensor(self.N, device=device))
        N = torch.tensor(self.N, device=device)
        sqrt2 = torch.sqrt(torch.tensor(2, device=device))
        H = lambda x: torch.erfc(x/sqrt2) / 2

        # initialize
        m_i2nu = torch.rand(size=(self.N, self.P), device=device) * 2 - 1
        u_mu2i = torch.rand(size=(self.P, self.N), device=device) * 2 - 1
        w_mu2i = torch.zeros(size=(self.P, self.N), device=device)
        sigma_mu2i = torch.zeros(size=(self.P, self.N), device=device)
        m_i = torch.zeros(self.N, device=device)
        xi = torch.from_numpy(self.xi).to(device)

        for i in range(maxIteration):
            m_i2nu = torch.tanh(u_mu2i.sum(dim=0).reshape(-1, 1) - u_mu2i.T)
            w_mu2i = (torch.sum(m_i2nu.T*xi, dim=1)[:, None].reshape(-1, 1) - m_i2nu.T*xi) / sqrtN
            sigma_mu2i = (torch.sum(1-m_i2nu*m_i2nu, dim=0)[:, None].reshape(-1, 1) - torch.t(1-m_i2nu*m_i2nu)) / N
            u_mu2i = (torch.log(H(-(xi/sqrtN + w_mu2i) / (torch.sqrt(sigma_mu2i) + 1e-16)) + 1e-16) - torch.log(H(-(-xi/sqrtN + w_mu2i) / (torch.sqrt(sigma_mu2i) + 1e-16)) + 1e-16)) / 2

            last_mi = m_i
            m_i = torch.tanh(torch.sum(u_mu2i, dim=0))
            delta_mi = torch.max(torch.abs(last_mi-m_i)).cpu()
            qLog.weakInfo('N: {}, alpha: {:.4f}, delta_mi: {}, iteration: {}'.format(self.N, self.alpha, delta_mi, i))
            
            if (delta_mi < eps):
                return True, i+1
            if (delta_mi > 1 and i >= 200):
                return False, i+1

        return False, maxIteration

    def replicaSymmetric(self):
        # define constant number & function
        sqrt2 = np.sqrt(2)
        sqrt2pi = np.sqrt(2*np.pi)
        G = lambda x: np.exp(-x*x/2) / sqrt2pi
        H = lambda x: special.erfc(x/sqrt2) / 2
        Z = lambda q, z: np.sqrt(q/(1-q+ 1e-10)) * z

        # calculate q & q^{\hat} & alpha
        alpha = 1.0
        maxIteration = 100 # calc q & q^{\hat}
        numEpoch = 200 # calc alha
        eps = 1e-5
        eta = 0.01

        for t in range(numEpoch):
            qLog.weakInfo('Epoch: {}, alpha: {}'.format(t, alpha))
            # fixed point method to calc q
            # q, q_hat = 0.5, 0.5
            q, q_hat = np.random.uniform(0.0, 1.0, size=2)
            last_q = q
            for _ in range(maxIteration):
                q = integrate.quad(lambda z: G(z) * np.power(np.tanh(np.sqrt(q_hat+ 1e-10)*z), 2), -np.inf, np.inf)[0]
                q_hat = alpha/(1-q+ 1e-10) * integrate.quad(lambda z: G(z) * np.power(G(-np.sqrt(q/(1-q+ 1e-10)+ 1e-10) * z) / (H(-np.sqrt(q/(1-q+ 1e-10)+ 1e-10) * z) + 1e-10), 2), -np.inf, np.inf)[0]

                if np.fabs(last_q - q) < eps:
                    break
                last_q = q
            
            # gradient decent to update alpha
            optimizeSech = lambda x: 2 * np.exp(-np.fabs(x)) / (1 + np.exp(-2*np.fabs(x)))
            tmp = integrate.quad(lambda z: G(z)*np.power(G(Z(q, z)) / (H(Z(q, z))+ 1e-10) * (Z(q, z) - G(Z(q, z)) / (H(Z(q, z))+ 1e-10)), 2), -np.inf, np.inf)[0] * integrate.quad(lambda z: G(z) * (np.power(optimizeSech(np.sqrt(q_hat)*z+ 1e-10), 4)+ 1e-10), -np.inf, np.inf)[0] / ((1-q)*(1-q))
            alpha -= eta * (alpha * tmp - 1) * tmp
        print('')
        

        






