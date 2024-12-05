import torch
import torch.nn as nn
import torch.nn.functional as F

class PassageRankingNN(nn.Module):
    """
    Neural Network for passage ranking using query and passage embeddings.
    """

    def __init__(self, embedding_dim):
        """
        Initializes the neural network.

        Args:
            embedding_dim (int): Dimension of the input embeddings.
        """
        super(PassageRankingNN, self).__init__()
        self.fc1 = nn.Linear(embedding_dim * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query_embed, passage_embed):
        """
        Forward pass of the neural network.

        Args:
            query_embed (Tensor): Query embeddings of shape (batch_size, embedding_dim).
            passage_embed (Tensor): Passage embeddings of shape (batch_size, embedding_dim).

        Returns:
            Tensor: Predicted relevancy scores of shape (batch_size,).
        """
        combined_embed = torch.cat((query_embed, passage_embed), dim=1)
        x = F.relu(self.fc1(combined_embed))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x)).squeeze()
        return x

def train_model(model, train_loader, optimizer, criterion, device, epochs=3):
    """
    Trains the neural network.

    Args:
        model (nn.Module): Neural network model.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to use for training.
        epochs (int): Number of training epochs.
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            queries, passages, relevancies = (
                batch['query'].to(device),
                batch['passage'].to(device),
                batch['relevancy'].to(device)
            )

            optimizer.zero_grad()
            outputs = model(queries, passages)
            loss = criterion(outputs, relevancies.squeeze())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

def evaluate_model(model, data_loader, device):
    """
    Evaluates the model on a dataset.

    Args:
        model (nn.Module): Neural network model.
        data_loader (DataLoader): DataLoader for evaluation data.
        device (torch.device): Device to use for evaluation.

    Returns:
        tuple: Mean average precision (MAP) and normalized discounted cumulative gain (NDCG).
    """
    model.eval()
    all_scores, all_true_labels, all_qids = [], [], []

    with torch.no_grad():
        for batch in data_loader:
            queries, passages, relevancies = (
                batch['query'].to(device),
                batch['passage'].to(device),
                batch['relevancy'].to(device)
            )
            scores = model(queries, passages).squeeze()
            all_scores.append(scores.cpu())
            all_true_labels.append(relevancies.squeeze().cpu())

    return torch.cat(all_scores), torch.cat(all_true_labels)
