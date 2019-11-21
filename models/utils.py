import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn


def loss_batch(model, loss_func, xb, yb, opt=None):
    """Calculate the loss, its gradient, and updates model parameters.
    
    It will always return the loss value of the batch evaluated on the model.
    If given an optimizer, it calculates the gradient of the loss over the
    model parameters (weights) and steps their values according to the 
    optimizer.

    Attributes:
        model (torch.nn.Module): model to evaluate upon
        loss_func (function): function that calculates loss over a batch
        xb (torch.tensor): batch of model inputs
        yb (torch.tensor): batch of model labels
        opt (torch.optim.Optimizer): optimizer to use for model stepping
    
    Returns:
        list: first element is loss evaluated over the batch, second item
           is number of elements in the batch (for averaging later on)

    """
    loss = loss_func(model(xb), yb)

    # If an optimizer is used, then run as if trianing, otherwise as if testing
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def acc_batch(model, xb, yb):
    """Calculate the model accuracy of a classifier over a batch.

    Accuracy is defined as true_results/number_of_cases

    Attributes:
        model (torch.nn.Module): model to evaluate upon
        xb (torch.tensor): batch of model inputs
        yb (torch.tensor): batch of model labels
    
    Returns:
        list: first element is accuracy evaluated over the batch, second 
            item is number of elements in the batch (for averaging later on)

    """
    max_vals, max_indices = torch.max(model(xb), 1)  # Maximum along output dimension (0 is batch dimension)
    accuracy = float((max_indices == yb).sum().float()/len(yb))
    return accuracy, len(yb)


def fit(epochs, model, loss_func, opt, train_dl, test_dl, verbose=True):
    """Trains a model over a given number of epochs.
    
    It optimizes for minimum value of the loss function of the model 
    evaluated over the train_dl training set, according to the optimizer,
    and also calculates loss results over the test set at each iteration.
    
    Attributes:
        epochs (int): number of times the model will train over the training
            set (it will see each example there epochs number of times)
        model (torch.nn.Module): model to be trained. Its weights will be
            changed in the optimization
        loss_func (function): function that applies the loss criterion over
            a batch of data on the model
        opt (torch.optim.Optimizer): optimizer to use for model stepping
        train_dl (torch.utils.data.DataLoader): provides an iterable over
            the training dataset, including inputs and their labels
        test_dl (torch.utils.data.DataLoader): provides an iterable over
            the testing dataset, including inputs and their labels
        verbose (bool, optional): determines whether to output progress
            in the console.
            
    Returns:
        list: list of lists with records of epochs, training loss, test
            loss and test accuracy
        
    """
    # Initialize the lists were progress will be recorded
    epoch_record, train_loss_record, test_loss_record, test_acc_record = [], [], [], []
    
    # For each epoch, train the model over the whole dataset in batches
    for epoch in range(epochs):
        
        # Training
        model.train()  # Sets the model to training mode
        train_losses, nums = zip(  
            *[loss_batch(model, loss_func, xb, yb, opt) for xb, yb in train_dl]
            )  # Evaluates the cross entropy for each batch and performs backpropagation and step
        train_loss = np.sum(np.multiply(train_losses, nums)) / np.sum(nums)  # Average the individual losses of batches

        # Evaluation
        model.eval()  # Sets the model to evaluation mode (would activate dropout and batchnorm)
        with torch.no_grad():  # This will deactivate the autograd engine and save memory
            test_losses, n_loss = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in test_dl]
                )  # Evaluates the cross entropy for each test batch, but does not backpropagate
            test_accs, n_acc = zip(*[acc_batch(model, xb, yb) for xb, yb in test_dl])  # Calculates accuracy as well
            
        test_loss = np.sum(np.multiply(test_losses, n_loss)) / np.sum(n_loss)  # Average the individual losses in the batch
        test_acc  = np.sum(np.multiply(test_accs,    n_acc)) / np.sum(n_acc )  # Average the individual accuracies in the batch
        
        # Record results for this epoch
        epoch_record.append(epoch)
        train_loss_record.append(train_loss)
        test_loss_record.append(test_loss)
        test_acc_record.append(test_acc)
        
        # Print results for this epoch
        if verbose:
            print(f"Epoch: {(epoch+1):3}    Train loss: {train_loss:7.5f}    Test loss: {test_loss:7.5f}    Test accuracy: {test_acc: 5.3f}")
    
    return epoch_record, train_loss_record, test_loss_record, test_acc_record


