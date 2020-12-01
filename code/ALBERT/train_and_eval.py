import sys
sys.path.append('../')
from sklearn.metrics import accuracy_score, roc_curve, auc

from transformers import adamw, get_linear_schedule_with_warmup
import pickle
import random
import time
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as f
import sys
import matplotlib.pyplot as plt
from albert_classifier import albertclassifier

def set_seed(seed_value=42):
    """set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def getdataloaders(batch_size):
    try:
        train_dataloader = torch.load("../../../dataloaders/albert/train_dataloader_"+str(batch_size)+".pth")
        val_dataloader = torch.load("../../../dataloaders/albert/val_dataloader_"+str(batch_size)+".pth")
        test_dataloader = torch.load("../../../dataloaders/albert/test_dataloader_"+str(batch_size)+".pth")
        return train_dataloader,val_dataloader,test_dataloader
    except:
        print("dataloaders have not been generated!")
        sys.exit()

def initialize(epochs=3,batch_size=16,lr=3e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    albert_classifier = albertclassifier().to(device)
    optimizer = adamw(albert_classifier.parameters(),lr=lr,eps=1e-8)
    train_dataloader,val_dataloaders,_ = getdataloaders(batch_size)
    num_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=num_steps)    
    return albert_classifier, optimizer, scheduler,train_dataloader,val_dataloader

def train(model, train_dataloader, val_dataloader=none, epochs=3, lr=3e-5, batch_size=16):
    train_loss,val_loss,val_accuracies = [],[],[]
    
    for epoch_i in range(epochs):
        total_sample_loss,sample_counter = 0,0
        print(f"{'epoch':^7} | {'batch':^7} | {'train loss':^12} | {'val loss':^10} | {'val acc':^9} | {'elapsed':^9}")
        print("-"*70)
        # measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()
        # reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        # put the model into the training mode        
        # for each batch of training data...       
        num_batches = len(train_dataloader)
        model.train()
        for step, batch in enumerate(train_dataloader):            
            sample_counter+= batch_size
            batch_counts +=1
            # load batch to gpu
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            # zero out any previously calculated gradients
            model.zero_grad()
            # perform a forward pass. this will return logits.
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
                print("now calculating validation loss and accuracy...")
                curr_val_loss, curr_val_accuracy = evaluate(model, val_dataloader)
                val_loss.append(curr_val_loss)
                print("train loss is ",total_sample_loss/16000," and validation loss is ",curr_val_loss)
                train_loss.append(total_sample_loss/16000)
                total_sample_loss = 0
                val_accuracies.append(curr_val_accuracy)
                model.train()
        avg_train_loss = total_loss / num_batches
        print("-"*70)
        curr_val_loss, curr_val_accuracy = evaluate(model, val_dataloader)
        train_loss.append(total_sample_loss/(sample_counter%16000))
        print("final train loss after epoch "+str(epoch_i)+" is ",total_sample_loss/(sample_counter%16000)," and validation loss is ",curr_val_loss)
        # print performance over the entire training data
        time_elapsed = time.time() - t0_epoch            
        print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {curr_val_loss:^10.6f} | {curr_val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
        print("-"*70)
        print("\n")
    
    model_train_results = {"train_loss":train_loss,"val_loss":val_loss,"val_accuracies":val_accuracies}

    with open("../../../trained_models/albert/"+"albert_"+str(size)+"_"+str(int(lr*(1e5)))+"_trainresults.pkl","wb") as f:
        pickle.dump(model_train_results,f)
    filename = "albert_trained_"+str(size)+"_"+str(int(lr*(1e5)))+"e-5.pth"
    torch.save(model,"../../../trained_models/albert/"+filename)
    print("training complete!")

def evaluate(model, val_dataloader):
    """after the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # put the model into the evaluation mode. the dropout layers are disabled during
    # the test time.
    model.eval()
    # tracking variables
    val_accuracy = []
    val_loss = []
    # for each batch in our validation set...
    for batch in val_dataloader:
        # load batch to gpu
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
        # compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        # compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())
        # get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        # calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)
    # compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)
    return val_loss, val_accuracy

def predict(model,dataloader):
    model.eval()
    all_logits = []
    print("fitting model on test data")
    
    for batch in dataloader:
        # load batch to gpu       
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]
        # compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)    
    # concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)
    # apply softmax to calculate probabilities
    probs = f.softmax(all_logits, dim=1).cpu().numpy()
    return probs

def plot_roc(probs,y_true,size,epochs,lr):
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'auc: {roc_auc:.4f}')       
    # get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'accuracy: {accuracy*100:.2f}%')    
    # plot roc auc
    figure, ax = plt.subplots()
    ax.set_title('receiver operating characteristic')
    ax.plot(fpr, tpr, 'b', label = 'auc = %0.2f' % roc_auc)
    ax.legend(loc = 'lower right')
    ax.plot([0, 1], [0, 1],'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('true positive rate')
    ax.set_xlabel('false positive rate')
    filename = "albert_roc_test_"+str(size)+"_"+str(epochs)+"_"+str(int(lr*(1e5)))+"e-5.jpg"
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
    dataframe = pd.dataframe.from_dict(report_data)
    filename = "ALBERT_report_"+str(size)+"_"+str(epochs)+"_"+str(int(lr*(1e5)))+".csv"
    dataframe.to_csv(filename, index = false)

if __name__ == '__main__':
    set_seed(1)    # set seed for reproducibility
    params_dict = {'batch_size':(32,64),'learning_rates':(5e-5,2e-5)}
    epochs=4
    loss_fn = nn.crossentropyloss()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    with open('../../../dataloaders/all_data.pkl','rb') as f:
        all_data = pickle.load(f)
    for size in params_dict['batch_size']:
        for lr in params_dict['learning_rates']:                              
            try:
                train_dataloader,val_dataloader,test_dataloader = getdataloaders(size) 
                filename = "albert_trained_"+str(size)+"_"+str(int(lr*(1e5)))+"e-5.pth"
                model = torch.load("../../../trained_models/albert/"+filename) 
                probs = predict(model,test_dataloader)
                plot_roc(probs, all_data['y_test'],size,epochs,lr)
                classification_report_csv(report,size,epochs,lr)
                print("roc plots generated")
            except oserror:
                print("model not found. starting the training...")
                torch.cuda.empty_cache()
                bert_classifier, optimizer, scheduler,train_dataloader,val_dataloader = initialize(epochs=epochs,batch_size=size,lr=lr)
                train(bert_classifier, train_dataloader, val_dataloader, epochs=4, lr=lr, batch_size=size)
