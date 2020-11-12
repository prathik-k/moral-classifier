import sys
sys.path.append('../')
from sklearn.metrics import accuracy_score, roc_curve, auc

from transformers import AdamW, get_linear_schedule_with_warmup
import pickle
import random
import time
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
from BERT_large_Classifier import BertLargeClassifier

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def getDataloaders(batch_size):
    try:
        train_dataloader = torch.load("../../../dataloaders/BERT-large/train_dataloader_"+str(batch_size)+".pth")
        val_dataloader = torch.load("../../../dataloaders/BERT-large/val_dataloader_"+str(batch_size)+".pth")
        test_dataloader = torch.load("../../../dataloaders/BERT-large/train_dataloader_"+str(batch_size)+".pth")
        return train_dataloader,val_dataloader,test_dataloader
    except:
        print("Dataloaders have not been generated!")
        sys.exit()

def initialize(epochs=3,batch_size=16,lr=3e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_classifier = BertLargeClassifier().to(device)
    optimizer = AdamW(bert_classifier.parameters(),lr=lr,eps=1e-8)
    train_dataloader,val_dataloaders,_ = getDataloaders(batch_size)
    num_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=num_steps)    
    return bert_classifier, optimizer, scheduler,train_dataloader,val_dataloader

def train(model, train_dataloader, val_dataloader=None, epochs=3, lr=3e-5, batch_size=16, evaluation=False):
    for epoch_i in range(epochs):
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        # Put the model into the training mode
        model.train()
        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            # Zero out any previously calculated gradients
            model.zero_grad()
            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if (step % 40 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
        avg_train_loss = total_loss / len(train_dataloader)
        print("-"*70)
        if evaluation == True:
            val_loss, val_accuracy = evaluate(model, val_dataloader)
            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    filename = "BERT-large_trained_"+str(size)+"_"+str(epochs)+"_"+str(int(lr*(1e5)))+"e-5.pth"
    torch.save(model,"../../../trained_models/BERT-large/"+filename)
    print("Training complete!")   

def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()
    # Tracking variables
    val_accuracy = []
    val_loss = []
    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())
        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    return val_loss, val_accuracy

def predict(model,dataloader):
    model.eval()
    all_logits = []
    print("Fitting model on validation data")
    for batch in dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()
    return probs

def plot_roc(probs,y_true,size,epochs,lr):
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')       
    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')    
    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    filename = "BERT-large_roc_"+str(size)+"_"+str(epochs)+"_"+str(int(lr*(1e5)))+"e-5.jpg"
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    set_seed(1)    # Set seed for reproducibility
    params_dict = {'num_epochs':(3,4),'batch_size':(16,8),'learning_rates':(5e-5,2e-5)}
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open('../../../dataloaders/all_data.pkl','rb') as file:
        all_data = pickle.load(file)
    
    for epochs in params_dict['num_epochs']:
        for size in params_dict['batch_size']:
            for lr in params_dict['learning_rates']:                              
                try:
                    train_dataloader,val_dataloader,test_dataloader = getDataloaders(size) 
                    filename = "BERT-large_trained_"+str(size)+"_"+str(epochs)+"_"+str(int(lr*(1e5)))+".pth"
                    model = torch.load("../../../trained_models/BERT-large"+filename)                    
                    probs = predict(model,val_dataloader)
                    plot_roc(probs, all_data['y_val'],size,epochs,lr)
                    print("ROC plots generated")
                except OSError:
                    print("Model not found. Starting the training...")
                    torch.cuda.empty_cache()
                    bert_classifier, optimizer, scheduler,train_dataloader,val_dataloader = initialize(epochs=epochs,batch_size=size,lr=lr)
                    train(bert_classifier, train_dataloader, val_dataloader, epochs=epochs, lr=lr, batch_size=size, evaluation=True)
