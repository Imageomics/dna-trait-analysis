import torch
import torch.nn.functional as F


def forward_step(
    model,
    batch,
    optimizer,
    is_train=True,
    return_diff=False,
    return_output=False,
    return_y=False,
):
    data, pca = batch
    data = data.cuda()
    pca = pca.cuda()
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

        return_items = [loss.item(), rmse]

        if return_diff:
            with torch.no_grad():
                diff = (out - pca).abs().sum(-1)
                best_diff = min(diff).item()
                worst_diff = max(diff).item()
                return_items.extend([best_diff, worst_diff])

        if return_output:
            return_items.append(out.cpu().numpy())

        if return_y:
            return_items.append(pca.cpu().numpy())

    return return_items
