import torch


def accuracy(outputs, labels):
    """ Computes the accuracy of a batch of predictions.
    Args:
        preds: (batch_size, num_classes) tensor of logits
        labels: (batch_size) tensor of labels
    Returns:
        accuracy: scalar tensor of accuracy
    """
    _, preds = torch.max(outputs, dim=1)
    print(labels, preds)
    return torch.mean((preds == labels).float())




