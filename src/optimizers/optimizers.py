import torch
import torch.nn as nn


# ============================================================================
# MUON OPTIMIZER
# ============================================================================

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G"""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X


class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1)) ** 0.5)


# ============================================================================
# LION OPTIMIZER
# ============================================================================

class Lion(torch.optim.Optimizer):
    """Lion optimizer (EvoLved Sign Momentum)"""
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                # Update = sign(momentum * exp_avg + (1-momentum) * grad)
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-group['lr'])

                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])

                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss


# ============================================================================
# SOPHIA OPTIMIZER (simplified)
# ============================================================================

class Sophia(torch.optim.Optimizer):
    """Sophia optimizer (Second-order Clipped Stochastic Optimization)"""
    def __init__(self, params, lr=1e-3, betas=(0.965, 0.99), rho=0.04, weight_decay=1e-1):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian'] = torch.zeros_like(p)

                exp_avg, hessian = state['exp_avg'], state['hessian']
                beta1, beta2 = group['betas']
                state['step'] += 1

                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Estimate Hessian diagonal (simplified - using grad squared)
                hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Clipped update
                h_clipped = hessian.clamp(min=1e-8)
                update = exp_avg / h_clipped.sqrt()
                update = torch.clamp(update, -group['rho'], group['rho'])

                # Apply update with weight decay
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['lr'] * group['weight_decay'])
                p.add_(update, alpha=-group['lr'])

        return loss


# ============================================================================
# OPTIMIZER SETUP
# ============================================================================

def setup_optimizer(model: nn.Module, config):
    """
    Setup optimizer based on config
    Returns list of optimizers (for compatibility with multi-optimizer setups)
    """
    optimizer_name = config.optimizer.lower()

    # Handle both 'lr' (TrainingConfig) and 'learning_rate' (SFTConfig/RLHFConfig)
    lr = getattr(config, 'lr', None) or getattr(config, 'learning_rate', 3e-4)

    if optimizer_name == "muon":
        # Muon for 2D params, AdamW for rest
        muon_params = []
        adamw_params = []

        for name, param in model.named_parameters():
            if (param.ndim == 2 and
                'token_embedding' not in name and
                'norm' not in name and
                param.requires_grad):
                muon_params.append(param)
            else:
                adamw_params.append(param)

        muon_optimizer = Muon(muon_params, lr=lr, momentum=config.muon_momentum, nesterov=config.muon_nesterov)
        adamw_optimizer = torch.optim.AdamW(adamw_params, lr=lr * 0.1, weight_decay=config.weight_decay)

        return [muon_optimizer, adamw_optimizer]

    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(config.adamw_beta1, config.adamw_beta2),
            eps=config.adamw_eps,
            weight_decay=config.weight_decay
        )
        return [optimizer]

    elif optimizer_name == "lion":
        optimizer = Lion(model.parameters(), lr=lr, betas=(config.lion_beta1, config.lion_beta2), weight_decay=config.weight_decay)
        return [optimizer]

    elif optimizer_name == "sophia":
        optimizer = Sophia(
            model.parameters(),
            lr=lr,
            betas=(config.sophia_beta1, config.sophia_beta2),
            rho=config.sophia_rho,
            weight_decay=config.weight_decay
        )
        return [optimizer]

    elif optimizer_name == "adafactor":
        try:
            from transformers.optimization import Adafactor
            optimizer = Adafactor(model.parameters(), lr=lr, weight_decay=config.weight_decay, scale_parameter=False, relative_step=False)
            return [optimizer]
        except ImportError:
            print("Adafactor requires transformers library. Falling back to AdamW.")
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
            return [optimizer]

    else:
        print(f"Unknown optimizer: {optimizer_name}. Falling back to AdamW.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.weight_decay)
        return [optimizer]


OPTIMIZER_NAMES = ["adamw", "muon", "lion", "sophia", "adafactor"]
