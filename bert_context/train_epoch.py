import torch
import torch.nn.functional as F
from tqdm import trange
import numpy as np
import time

def train(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader=None, epochs=4, evaluation=False, device='cpu', FINETUNE=False):
    """Train the BertClassifier model.
    """
    print("Start training...\n")
    train_losses = []
    val_losses = []
    val_acc = []
    for epoch_i in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-" * 70)
        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts += 1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()

            logits = model(b_input_ids, b_attn_mask)

            if FINETUNE:
                logits = logits["logits"]

            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (step % 200 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch

                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                train_losses.append(batch_loss / batch_counts)

                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)


        print("-" * 70)
        if evaluation == True:
            val_loss, val_accuracy = evaluate(model, loss_fn, val_dataloader, device, FINETUNE)
            val_losses.append(val_loss)
            val_acc.append(val_accuracy / 100)
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-" * 70)
        print("\n")
    print("Training complete!")
    return train_losses, val_losses, val_acc


def evaluate(model, loss_fn, val_dataloader, device, FINETUNE=False):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    model.eval()
    val_accuracy = []
    val_loss = []

    for batch in val_dataloader:
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        if FINETUNE:
            logits = logits["logits"]
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        preds = torch.argmax(logits, dim=1).flatten()

        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


def bert_predict(model, b_input_ids, b_attn_mask, FINETUNE):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    label = ['entailment', 'neutral', 'contradiction']
    model.eval()
    with torch.no_grad():
        logits = model(b_input_ids, b_attn_mask)
        if FINETUNE:
            logits = logits["logits"]
    preds = torch.argmax(logits, dim=1).flatten()
    return label[preds]