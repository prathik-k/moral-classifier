import sys
sys.path.append('../')
from sklearn.metrics import accuracy_score, roc_curve, auc,classification_report

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
from ALBERT_Classifier import AlbertClassifier

def set_seed(seed_value=42):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def getDataloaders(batch_size):
    try:
        train_dataloader = torch.load("../../../dataloaders/ALBERT/train_dataloader_"+str(batch_size)+".pth")
        val_dataloader = torch.load("../../../dataloaders/ALBERT/val_dataloader_"+str(batch_size)+".pth")
        test_dataloader = torch.load("../../../dataloaders/ALBERT/test_dataloader_"+str(batch_size)+".pth")
        return train_dataloader,val_dataloader,test_dataloader
    except:
        print("Dataloaders have not been generated!")
        sys.exit()

def initialize(epochs=3,batch_size=16,lr=3e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_classifier = BertClassifier().to(device)
    optimizer = AdamW(bert_classifier.parameters(),lr=lr,eps=1e-8)
    train_dataloader,val_dataloaders,_ = getDataloaders(batch_size)
    num_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=num_steps)    
    return bert_classifier, optimizer, scheduler,train_dataloader,val_dataloader

def train(model, train_dataloader, val_dataloader=None, epochs=3, lr=3e-5, batch_size=16):
    train_loss,val_loss,val_accuracies = [],[],[]
    
    for epoch_i in range(epochs):
        total_sample_loss,sample_counter = 0,0
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)
        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()
        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        # Put the model into the training mode        
        # For each batch of training data...       
        num_batches = len(train_dataloader)
        model.train()
        for step, batch in enumerate(train_dataloader):            
            sample_counter+= batch_size
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
            total_sample_loss+=loss.item()*batch_size
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if (step % 40 == 0 and step != 0) or (step == num_batches - 1):
                time_elapsed = time.time() - t0_batch
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")
                batch_loss, batch_counts = 0, 0                
                t0_batch = time.time()            
            if sample_counter%16000==0:
                print("Now calculating validation loss and accuracy...")
                curr_val_loss, curr_val_accuracy = evaluate(model, val_dataloader)
                val_loss.append(curr_val_loss)
                print("Train loss is ",total_sample_loss/16000," and validation loss is ",curr_val_loss)
                train_loss.append(total_sample_loss/16000)
                total_sample_loss = 0
                val_accuracies.append(curr_val_accuracy)
                model.train()
        avg_train_loss = total_loss / num_batches
        print("-"*70)
        curr_val_loss, curr_val_accuracy = evaluate(model, val_dataloader)
        train_loss.append(total_sample_loss/(sample_counter%16000))
        print("Final train loss after epoch "+str(epoch_i)+" is ",total_sample_loss/(sample_counter%16000)," and validation loss is ",curr_val_loss)
        # Print performance over the entire training data
        time_elapsed = time.time() - t0_epoch            
        print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {curr_val_loss:^10.6f} | {curr_val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
        print("-"*70)
        print("\n")
    
    model_train_results = {"train_loss":train_loss,"val_loss":val_loss,"val_accuracies":val_accuracies}

    with open("../../../trained_models/ALBERT/"+"ALBERT_"+str(size)+"_"+str(int(lr*(1e5)))+"_trainResults.pkl","wb") as f:
        pickle.dump(model_train_results,f)
    filename = "ALBERT_trained_"+str(size)+"_"+str(int(lr*(1e5)))+"e-5.pth"
    torch.save(model,"../../../trained_models/ALBERT/"+filename)
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
    print("Fitting model on test data")
    
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
    figure, ax = plt.subplots()
    ax.set_title('Receiver Operating Characteristic')
    ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    filename = "ALBERT_roc_test_"+str(size)+"_"+str(epochs)+"_"+str(int(lr*(1e5)))+"e-5.jpg"
    figure.savefig(filename)
    plt.close(figure)

def classification_report_csv(report,size,epochs,lr):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    filename = "ALBERT_report_"+str(size)+"_"+str(epochs)+"_"+str(int(lr*(1e5)))+".csv"
    dataframe.to_csv(filename, index = False)    

if __name__ == '__main__':
    set_seed(1)    # Set seed for reproducibility
    epochs=4
    params_dict = {'batch_size':(32,64),'learning_rates':(5e-5,2e-5)}
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open('../../../dataloaders/all_data.pkl','rb') as f:
        all_data = pickle.load(f)
    for size in params_dict['batch_size']:
        for lr in params_dict['learning_rates']:                              
            try:
                train_dataloader,val_dataloader,test_dataloader = getDataloaders(size) 
                filename = "ALBERT_trained_"+str(size)+"_"+str(int(lr*(1e5)))+"e-5.pth"
                model = torch.load("../../../trained_models/ALBERT/"+filename) 
                probs = predict(model,test_dataloader)
                y_pred = probs[:, 1]
                print(y_pred,all_data['y_test'])
                plot_roc(probs, all_data['y_test'],size,epochs,lr)
                report = classification_report(all_data['y_test'], y_pred)
                classification_report_csv(report,size,epochs,lr)
                print("ROC plots generated")
            except OSError:
                print("Model not found. Starting the training...")
                torch.cuda.empty_cache()
                bert_classifier, optimizer, scheduler,train_dataloader,val_dataloader = initialize(epochs=4,batch_size=size,lr=lr)
                train(bert_classifier, train_dataloader, val_dataloader, epochs=4, lr=lr, batch_size=size)
