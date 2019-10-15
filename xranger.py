#Ranger deep learning optimizer - RAdam + Lookahead combined.
#https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

#Ranger has now been used to capture 12 records on the FastAI leaderboard.

#This version = 9.3.19  

#Credits:
#RAdam -->  https://github.com/LiyuanLucasLiu/RAdam
#Lookahead --> rewritten by lessw2020, but big thanks to Github @LonePatient and @RWightman for ideas from their code.
#Lookahead paper --> MZhang,G Hinton  https://arxiv.org/abs/1907.08610

#summary of changes: 
#full code integration with all updates at param level instead of group, moves slow weights into state dict (from generic weights), 
#supports group learning rates (thanks @SHolderbach), fixes sporadic load from saved model issues.
#changes 8/31/19 - fix references to *self*.N_sma_threshold; 
                #changed eps to 1e-5 as better default than 1e-8.

import math
import torch
from torch.optim.optimizer import Optimizer, required
import itertools as it



class XRanger(Optimizer):

    def __init__(self, params, lr=1e-3, alpha=0.5, k=6, N_sma_threshhold=5, betas=(.95,0.995), eps=1e-6, weight_decay=0):
        #parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas, N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params,defaults)

        self.N_sma_threshhold = N_sma_threshhold

        self.alpha = alpha
        self.k = k 

        self.radam_buffer = [[None,None,None] for ind in range(10)]
        
        self.p0 = set()
        self.p1 = set()


    def set_params_for_wd(self,p0, p1):
        self.p0 = set(p0)
        self.p1 = set(p1)

    def __setstate__(self, state):
        #print("set state called")
        super(XRanger, self).__setstate__(state)

    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']
            eps = group['eps']
            k = group['k']
            
            for p in group['params']:
                if p.grad is None: continue
                self._step_param(p, lr, beta1, beta2, wd, eps, k)

    def _step_param(self, p, lr, beta1, beta2, wd, eps, k):
        grad = p.grad.data.float()
        if grad.is_sparse:
            raise RuntimeError('Ranger optimizer does not support sparse gradients')

        pd32 = p.data.float()

        state = self.state[p]  #get state dict for this param

        if len(state) == 0:   #if first time to run...init dictionary with our desired entries
            #if self.first_run_check==0:
                #self.first_run_check=1
                #print("Initializing slow buffer...should not see this at load from saved model!")
            state['step'] = 0
            state['exp_avg'] = torch.zeros_like(pd32)
            state['exp_avg_sq'] = torch.zeros_like(pd32)

            #look ahead weight storage now in state dict 
            state['slow_buffer'] = torch.empty_like(p.data)
            state['slow_buffer'].copy_(p.data)

        else:
            state['exp_avg'] = state['exp_avg'].type_as(pd32)
            state['exp_avg_sq'] = state['exp_avg_sq'].type_as(pd32)

        #begin computations 
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        
        #compute variance mov avg
        exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
        #compute mean moving avg
        exp_avg.mul_(beta1).add_(1 - beta1, grad)

        state['step'] += 1


        buffered = self.radam_buffer[int(state['step'] % 10)]
        if state['step'] == buffered[0]:
            N_sma, step_size = buffered[1], buffered[2]
        else:
            buffered[0] = state['step']
            beta2_t = beta2 ** state['step']
            N_sma_max = 2 / (1 - beta2) - 1
            N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
            buffered[1] = N_sma
            if N_sma > self.N_sma_threshhold:
                step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
            else:
                step_size = 1.0 / (1 - beta1 ** state['step'])
            buffered[2] = step_size

        if wd > 0:
            delta = lr*wd
            for w in self.p0:
                if p is w:
                    pd32.mul_(1-delta)
                    print(f"Decaying to 0 weight of shape {pd32.shape}")
            for w in self.p0:
                if p is w:
                    pd32.mul_(1-delta)
                    pd32.add_(delta)
                    print(f"Decaying to 1 weight of shape {pd32.shape}")

        if N_sma > self.N_sma_threshhold:
            denom = exp_avg_sq.sqrt().add_(eps)
            pd32.addcdiv_(-step_size * lr, exp_avg, denom)
        else:
            pd32.add_(-step_size * lr, exp_avg)

        p.data.copy_(pd32)

        #integrated look ahead...
        #we do it at the param level instead of group level
        if state['step'] % k == 0:
            slow_p = state['slow_buffer'] #get access to slow param tensor
            slow_p.add_(self.alpha, p.data - slow_p)  #(fast weights - slow weights) * alpha
            p.data.copy_(slow_p)  #copy interpolated weights to RAdam param tensor

    

