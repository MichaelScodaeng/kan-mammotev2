import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import csv
import os
from tqdm import tqdm
from LeTE import CombinedLeTE

class EventBasedMNIST(Dataset):
    def __init__(self, root, train=True, threshold=0.9, transform=None, download=True):
        self.root = root
        self.train = train
        self.threshold = threshold
        self.transform = transform
        
        self.data = datasets.MNIST(root=self.root, train=self.train, transform=self.transform, download=download)
        self.event_data = []
        self.labels = []
        for img, label in self.data:
            img_flat = img.view(-1)  # (784,)
            events = torch.nonzero(img_flat > self.threshold).squeeze()
            events = torch.sort(events).values
            self.event_data.append(events)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.event_data)
    
    def __getitem__(self, idx):
        return self.event_data[idx], self.labels[idx]

def custom_collate_fn(batch):
    events_list = []
    labels_list = []
    lengths = []
    for events, label in batch:
        events_list.append(events)
        labels_list.append(label)
        lengths.append(events.shape[0])
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    padded_events = pad_sequence(events_list, batch_first=True, padding_value=0)  # (batch, max_len)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return padded_events, lengths, labels_tensor

class LSTMClassifier(nn.Module):
    def __init__(self, input_size=784, embedding_dim=32, hidden_dim=128, num_classes=10):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        # Init LeTE
        self.time_encoder = CombinedLeTE(embedding_dim, p=0.5)
        
    def forward(self, x, lengths):
        embedded = self.time_encoder(x)  # (batch, seq_len, embedding_dim)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, c_n) = self.lstm(packed)
        h_n = h_n[-1]  # (batch, hidden_dim)
        out = self.fc(h_n)  # (batch, num_classes)
        return out

def evaluate(model, data_loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for padded_events, lengths, labels in data_loader:
            padded_events = padded_events.float().to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            outputs = model(padded_events, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc

if __name__ == "__main__":
    transform = transforms.ToTensor()
    train_dataset = EventBasedMNIST(root="./data", train=True, threshold=0.9, transform=transform, download=True)
    test_dataset = EventBasedMNIST(root="./data", train=False, threshold=0.9, transform=transform, download=True)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False, collate_fn=custom_collate_fn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(input_size=784, embedding_dim=32, hidden_dim=128, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 200
    accuracy_per_epoch = []
    loss_per_epoch = []
    test_accuracy_per_epoch = []
    test_loss_per_epoch = []

    log_file = "training_log_LeTE.csv"
    if os.path.exists(log_file):
        os.remove(log_file)

    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"])

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Training
        for batch_idx, (padded_events, lengths, labels) in enumerate(tqdm(train_loader)):
            padded_events = padded_events.float().to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(padded_events, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / total_samples
        acc = total_correct / total_samples
        accuracy_per_epoch.append(acc)
        loss_per_epoch.append(avg_loss)

        # Testing
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)
        test_accuracy_per_epoch.append(test_acc)
        test_loss_per_epoch.append(test_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] finished. "
            f"Train Loss: {avg_loss:.4f}, Train Accuracy: {acc:.4f}, "
            f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # Save results to csvs
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, avg_loss, acc, test_loss, test_acc])

    # Printing results
    print("Final Training Accuracy per epoch:", accuracy_per_epoch)
    print("Final Training Loss per epoch:", loss_per_epoch)
    print("Final Test Accuracy per epoch:", test_accuracy_per_epoch)
    print("Final Test Loss per epoch:", test_loss_per_epoch)
