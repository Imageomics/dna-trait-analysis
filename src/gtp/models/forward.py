import torch
import torch.nn.functional as F


def forward_step(
    model, batch, optimizer, out_dims, out_start_idx=0, is_train=True, return_diff=False
):
    data, pca = batch
    data = data.cuda()
    pca = pca[:, out_start_idx : out_start_idx + out_dims].cuda()
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

        if return_diff:
            with torch.no_grad():
                diff = (out - pca).abs().sum(-1)
                best_diff = min(diff).item()
                worst_diff = max(diff).item()
                return loss.item(), rmse, best_diff, worst_diff
    return loss.item(), rmse
