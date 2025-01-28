from torch import optim

def get_optimizer(model, optimizer_name, lr=0.001):
    params = [p for p in model.parameters() if p.requires_grad]
    
    if optimizer_name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=0.01)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    elif optimizer_name == 'nesterov':
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")