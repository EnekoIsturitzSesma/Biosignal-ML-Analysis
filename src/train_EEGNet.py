import torch
import torch.nn as nn

def apply_max_norm(model, max_val=1.0):
    for name, param in model.named_parameters():
        if 'weight' in name and param.ndim > 1:
            if 'TemporalConv' in name or 'DepthSpatialConv' in name or 'FC' in name:
                param.data.copy_(torch.renorm(param.data, p=2, dim=0, maxnorm=max_val))

def train_model(model, train_dl, val_dl, epochs=100, lr=0.0005, patience=20):

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0

    for epoch in range(epochs):

        model.train()
        train_loss = 0.0
        train_correct = 0

        for batch_x, batch_y in train_dl:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            apply_max_norm(model, max_val=1.0)

            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(output, 1)
            train_correct += (predicted == batch_y).sum().item()

        train_loss /= len(train_dl.dataset)
        train_acc = train_correct / len(train_dl.dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for batch_x, batch_y in val_dl:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                output = model(batch_x)
                loss = criterion(output, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(output, 1)
                val_correct += (predicted == batch_y).sum().item()

        val_loss /= len(val_dl.dataset)
        val_acc = val_correct / len(val_dl.dataset)
        scheduler.step(val_acc)

        print(f"Epoch {epoch+1}: "
              f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
              f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(next(model.parameters()).device), labels.to(next(model.parameters()).device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total