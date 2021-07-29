import torch as t
import numpy as np
from tqdm import tqdm
from lib.plot import plt_train

dtype=t.FloatTensor

def train_epoch(model, loss, loader, optimizer):
    losses = []
    for X, y in loader:
        X = X.type(dtype).cuda()
        y = y.cuda()
        prediction = model(X)
        loss_batch = loss(prediction, y)
        losses.append(loss_batch.item())
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()
    return losses

def eval_epoch(model, eval_loss, loader):
    losses = []
    for X, y in loader:
        X = X.type(dtype).cuda()
        prediction = model(X)
        ls = eval_loss(y.numpy(), np.where(prediction.cpu().detach().numpy() > 0.5, 1, 0))
        losses.append(ls)

    return losses

def train(model, train_loader, val_loader, epochs, optimizer, loss, eval_loss):

    train_loss_epochs = []
    eval_loss_epochs = []

    try:
        for epoch in tqdm(range(epochs)):
            model = model.train()
            losses_train = train_epoch(model, loss, train_loader, optimizer)
            train_loss_epochs.append(np.mean(losses_train))

            model = model.eval()
            losses_eval = eval_epoch(model, eval_loss, val_loader)
            eval_loss_epochs.append(np.mean(losses_eval))

            plt_train(epoch, train_loss_epochs, eval_loss_epochs)


    except KeyboardInterrupt:
        if tolerate_keyboard_interrupt:
            pass
        else:
            raise KeyboardInterrupt
    return eval_loss_epochs