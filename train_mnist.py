"""CYSE 635: MNIST CNN experiments. Run: python train_mnist.py --epochs 12"""
import argparse, copy, csv, json, random
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

class DigitCNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x, features=False):
        f1 = torch.relu(self.conv1(x))
        f2 = torch.relu(self.conv2(self.pool(f1)))
        x = self.pool(f2).flatten(1)
        x = self.dropout(torch.relu(self.fc1(x)))
        logits = self.fc2(x)
        return (logits, f1, f2) if features else logits

def evaluate(model, loader, device, criterion):
    model.eval(); loss_sum=0; correct=0; n=0; labels=[]; predictions=[]
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            out=model(x); loss_sum += criterion(out,y).item()*len(y)
            pred=out.argmax(1); correct+=(pred==y).sum().item(); n+=len(y)
            labels.extend(y.cpu().tolist()); predictions.extend(pred.cpu().tolist())
    return loss_sum/n, correct/n, labels, predictions

def run(name, dropout, lr, train_loader, val_loader, test_loader, args, device, outdir):
    model=DigitCNN(dropout).to(device); optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    criterion=nn.CrossEntropyLoss(); history=[]; best=float('inf'); patience=0; best_weights=None
    for epoch in range(1,args.epochs+1):
        model.train(); total=0; correct=0; n=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device); optimizer.zero_grad()
            logits=model(x); loss=criterion(logits,y); loss.backward(); optimizer.step()
            total+=loss.item()*len(y); correct+=(logits.argmax(1)==y).sum().item(); n+=len(y)
        vl,va,_,_=evaluate(model,val_loader,device,criterion)
        row=dict(epoch=epoch,train_loss=total/n,train_accuracy=correct/n,val_loss=vl,val_accuracy=va)
        history.append(row); print(name,row,flush=True)
        if vl < best-0.0001: best=vl; best_weights=copy.deepcopy(model.state_dict()); patience=0
        else: patience+=1
        if patience>=args.patience: break
    model.load_state_dict(best_weights)
    tl,ta,y_true,y_pred=evaluate(model,test_loader,device,criterion)
    torch.save(model.state_dict(),outdir/f'{name}.pt')
    with open(outdir/f'{name}_history.csv','w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=history[0].keys()); writer.writeheader();writer.writerows(history)
    fig,ax=plt.subplots(1,2,figsize=(10,4))
    for metric,axis in [('loss',ax[0]),('accuracy',ax[1])]:
        axis.plot([r['epoch'] for r in history],[r['train_'+metric] for r in history],label='Train')
        axis.plot([r['epoch'] for r in history],[r['val_'+metric] for r in history],label='Validation')
        axis.set(title=f'{name}: {metric}',xlabel='Epoch',ylabel=metric);axis.legend();axis.grid(alpha=.2)
    fig.tight_layout();fig.savefig(outdir/f'{name}_learning_curves.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,7));ConfusionMatrixDisplay(confusion_matrix(y_true,y_pred),display_labels=list(range(10))).plot(ax=ax,cmap='Blues',colorbar=False)
    fig.tight_layout();fig.savefig(outdir/f'{name}_confusion_matrix.png',dpi=160);plt.close(fig)
    report=classification_report(y_true,y_pred,output_dict=True,zero_division=0)
    with open(outdir/f'{name}_classification_report.json','w') as f:json.dump(report,f,indent=2)
    if name=='baseline':
        weights=model.conv1.weight.detach().cpu().numpy()
        fig,axes=plt.subplots(4,8,figsize=(10,5))
        for i,axis in enumerate(axes.flat):axis.imshow(weights[i,0],cmap='gray');axis.axis('off')
        fig.suptitle('First convolutional layer: 32 learned 3x3 filters');fig.tight_layout();fig.savefig(outdir/'learned_filters.png',dpi=160);plt.close(fig)
        image,_=next(iter(test_loader));image=image[:1].to(device)
        with torch.no_grad():_,f1,f2=model(image,features=True)
        fig,axes=plt.subplots(2,8,figsize=(12,4))
        for i in range(8):
            axes[0,i].imshow(f1[0,i].cpu(),cmap='viridis');axes[0,i].axis('off')
            axes[1,i].imshow(f2[0,i].cpu(),cmap='viridis');axes[1,i].axis('off')
        fig.suptitle('Feature maps: conv1 (top) and conv2 (bottom)');fig.tight_layout();fig.savefig(outdir/'feature_maps.png',dpi=160);plt.close(fig)
    return dict(name=name,dropout=dropout,learning_rate=lr,epochs_trained=len(history),test_loss=tl,test_accuracy=ta,precision_macro=report['macro avg']['precision'],recall_macro=report['macro avg']['recall'],f1_macro=report['macro avg']['f1-score'],best_validation_loss=best)

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=12);p.add_argument('--patience',type=int,default=3);p.add_argument('--batch-size',type=int,default=128);p.add_argument('--data-dir',default='./data');p.add_argument('--out-dir',default='./results');args=p.parse_args()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');print('Device:',device)
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    train_all=datasets.MNIST(args.data_dir,train=True,download=True,transform=transform)
    test=datasets.MNIST(args.data_dir,train=False,download=True,transform=transform)
    train,val=random_split(train_all,[50000,10000],generator=torch.Generator().manual_seed(42))
    loader=lambda data,shuffle=False:DataLoader(data,batch_size=args.batch_size,shuffle=shuffle,num_workers=0)
    train_loader,val_loader,test_loader=loader(train,True),loader(val),loader(test)
    outdir=Path(args.out_dir);outdir.mkdir(parents=True,exist_ok=True)
    results=[]
    for name,dropout,lr in [('baseline',0.3,0.001),('higher_dropout',0.5,0.001),('lower_learning_rate',0.3,0.0003)]:
        results.append(run(name,dropout,lr,train_loader,val_loader,test_loader,args,device,outdir))
    with open(outdir/'experiment_summary.json','w') as f:json.dump(results,f,indent=2)
    print('\nFINAL RESULTS:',json.dumps(results,indent=2))
if __name__=='__main__':main()
