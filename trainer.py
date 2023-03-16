import torch
import torch.nn as nn
import torch.optim as optim


class Trainer:
    def __init__(self, model, criterion, optimizer, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        return loss.item()

    def predict_step(self, inputs):
        self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(self.device)
            outputs = self.model(inputs)
        return outputs

    def train(self, train_loader, val_loader, num_epochs):
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            train_loss = 0.0
            for i, batch in enumerate(train_loader):
                loss = self.train_step(batch)
                train_loss += loss
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            val_loss = 0.0
            for i, batch in enumerate(val_loader):
                loss = self.eval_step(batch)
                val_loss += loss
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        return train_losses, val_losses
