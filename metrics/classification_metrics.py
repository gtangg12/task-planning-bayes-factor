import torch


def accuracy(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ Computes the accuracy of a batch of predictions.
    Args:
        preds: (num_data, num_classes) tensor of logits
        labels: (num_data) tensor of labels
    Returns:
        accuracy: scalar tensor of accuracy
    """
    return torch.mean((preds == labels).float())


def label_frequency(labels: torch.Tensor) -> torch.Tensor:
    """ Computes the frequency of each label in a batch.
    Args:
        labels: (num_data) tensor of labels
    Returns:
        freq: (num_classes) tensor where index i is the frequency of label i
    """
    return torch.bincount(labels)



