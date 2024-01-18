import torch
import torch.nn.functional as F

def forward_step(model, batch, optimizer, out_dims, is_train=True):
    name, data, pca = batch
    data = data.cuda()
    pca = pca[:, :out_dims].cuda()
    with torch.set_grad_enabled(is_train):
        if is_train:
            model.train()
        else:
            model.eval()
        out = model(data)
        loss = F.mse_loss(out, pca)
        rmse = torch.sqrt(loss).item()

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    return loss.item(), rmse