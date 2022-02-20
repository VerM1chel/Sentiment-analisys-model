import torch
from tqdm import tqdm

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets.unsqueeze(1))

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    for batchIndex, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = data['ids']
        mask = data['mask']
        targets = data['targets']

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids, mask)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

def eval_fn(dataloader, model, device):
    fin_outputs = []
    fin_targets = []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            ids = data['ids']
            mask = data['mask']
            targets = data['targets']

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids, mask)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
