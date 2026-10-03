import torch 
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,random_split, DataLoader


#%% 
torch.manual_seed(123)


n_samples =1000 
n_features = 5
X = torch.randn(n_samples,n_features)
X.shape

true_W = torch.tensor([
    [2.0],
    [-1.5],
    [1.0],
    [0.5],
    [-2.0]
])

true_b = 0.3

# Continuous score
score = X @ true_W + true_b

# Add some noise
score += 0.5 * torch.randn(n_samples, 1)

# --------------------------------------------------
# 3. Convert score into class labels
# --------------------------------------------------

y = (score > 0).float()
y # 0 1 1 1 1 10 1  0 1





dataset = TensorDataset(X,y)
dataset
N = len(dataset)
print(dataset)

n_train = int(.7 *N)
n_val = int (  (.15) *N    ) 
n_test = N-n_train-n_val


generator = torch.Generator().manual_seed(123) 

train_dataset,val_dataset,test_dataset = random_split(
    dataset, [n_train,n_val,n_test], generator=generator
)



print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

print(train_dataset) # this is not another huge size object ..its just have the indices of the original dataset ..




# create daataloaders 
batch_size = 32
train_loader  = DataLoader(train_dataset,batch_size=batch_size,shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size , shuffle=False )
test_loader = DataLoader(test_dataset, batch_size=batch_size , shuffle=False )

print(train_loader) # this is an dataloader object..not entire new copy of traindataset..it just redirecting to the original dataset . 



# for X,y in train_loader:


#     print(X.shape,y.shape)

def my_activation(x):
   return  torch.where( x>=0 ,x,.01*x) #leaky relu


class NoisyLinear(nn.Module):
    def __init__(self,in_features ,out_features , sigma = .01):
        super().__init__()
        self.linear  = nn.Linear(in_features=in_features, out_features=out_features)
        self.sigma = sigma 
    def forward(self,x):
        if self.training:
            noise =torch.randn_like(self.linear.weight)*self.sigma 
            noisy_weight = self.linear.weight + noise 

            return torch.nn.functional.linear(x,noisy_weight,self.linear.bias) # functional used so that we could use noisy_weight instead of linear.weight.without functional function , we will have an error
        
        

        else :
            return self.linear(x)


class NeuralNetwork(nn.Module): 

    def __init__(self):
        super().__init__()
        self.layer1 = NoisyLinear(5,16,sigma=.01)
        self.layer2 = nn.Linear (16,8)
        self.output = nn.Linear (8,1)
        self.dropout = nn.Dropout(p=.3)


    def forward (self,x): #small x because x is just for signle persons info
        # x = torch.sin(x)
        x = self.layer1(x)
        x = my_activation(x) 
        x= self.dropout(x)

        # x = torch.sin(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x =self.output(x) 
        return x 
    
model = NeuralNetwork() 
# model = model.to(device=device)
loss_fn = nn.BCEWithLogitsLoss()

def my_bce_loss(logits, targets,gamma = 2.0):
    
    probabilities = torch.sigmoid(logits)

    loss = (
        -targets * torch.log(probabilities)
        -(1 - targets) * torch.log(1 - probabilities)
    ) 

    loss = loss **gamma

    return loss.mean()





optimizer = torch.optim.Adam(model.parameters() ,lr=.01 , betas=[0.9,.999] , amsgrad=True ,eps=1e-8    )


print(model)

for name,param in model.named_parameters():
    print(f"Layer: {name:15} | Shape: {list(param.shape)} | Requires Grad: {param.requires_grad}")





max_epoch =100 
train_losses= [] 
val_losses = [] 

for epoch in range(max_epoch)  : 

    model.train() # its not trains ..it just set the internal flag , model.train flag =True.Used in dropouts 
    total_train_loss = 0 

    for X_batch,y_batch in train_loader: #as train_loader is pytorch object , so no new memory has been created for X_batch and y_batch .Rather its refrencing the original dataset


        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits,y_batch)  # this loss is the avegrage loss of all batched 
        loss.backward()
        optimizer.step()

        #accumulate loss
        total_train_loss+=loss.item() * X_batch.size(0) # size is method of X_batch object ..thats why use siz(0), not size[0]
        #AVG LOSS * batch size = loss for whole batch
    #avetrage total loss
    train_loss =  total_train_loss /  len(train_dataset)


    #validation 
    model.eval() # model is in evaluation mode now
    total_eval_loss = 0 



    with torch.no_grad():

        for X_batch,y_batch in val_loader :
            logits  =model(X_batch) 
            loss = loss_fn( logits , y_batch   )

            total_eval_loss += loss.item() * X_batch.size(0)
        val_loss = total_eval_loss / len(val_dataset)

        train_losses.append(train_loss)
        val_losses .append(val_loss) 

        print(



            f"epoch {epoch} | train loss = {train_loss} | val loss = {val_loss}"


        )



plt.plot(train_losses)
plt.plot(val_losses)
plt.legend(["training loss","validation loss"])

plt.show()