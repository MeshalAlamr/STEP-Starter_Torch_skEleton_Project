#TODO
#save model

#imports
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from model import BaseModel
from dataloader import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import datetime
import logging
from tqdm import tqdm


#args
parser = argparse.ArgumentParser(description='args for experement')
parser.add_argument('--personality_exp', type=bool, default=True, help='if True load csvs in Personality_Score')
parser.add_argument('--offline_logs', type=bool, default=True, help='if True print logs in terminal')
parser.add_argument('--online_logs', type=str, default='', help='if not empty use wandb to upload logs, provide wandb api key as input')
parser.add_argument('--save_path', type=str, default='Experements', help='path to save model')
parser.add_argument('--audio_length', type=int, default=80001, help='All audious will be either padded or truncated to the input')
parser.add_argument('--epoch', type=int, default=50, help='number of trainin epochs')
parser.add_argument('--exp_name', type=str, default= datetime.datetime.now(), help='name of experement on wandb')
parser.add_argument('--test_split', type=float, default=0.2, help='persentage (0.0 -> 1.0) of data to be allocated for test. Note: training split is infered.')
parser.add_argument('--batch', type=int, default=12, help='')
parser.add_argument('--shuffle', type=bool, default=False, help='')
parser.add_argument('--lr', type=float, default=1e-3, help='')
parser.add_argument('--embed_length', type=int, default=768, help='')
args = parser.parse_args()

#---------Check for args validity------------

#cretae save dir
if not args.save_path:
        raise ValueError(f'--save_path must be provided. current value is {args.save_path}')
else:
    
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    
    
    if not os.path.isdir(os.path.join(args.save_path, args.exp_name)):
        os.mkdir(os.path.join(args.save_path, args.exp_name))

#check test_split value
if args.test_split > 1 or args.test_split < 0.0:
    raise ValueError(f'--test_split must be between 0.0 and 1.0')

if not args.exp_name:
    raise ValueError('--exp_name must be provided')

if args.offline_logs:
    logging.basicConfig(filename=os.path.join(args.save_path, args.exp_name,'logs.log') , filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.INFO)

#check for online loging
if args.online_logs:
    if not args.exp_name:
        raise ValueError('--exp_name must be provided when --online_logs is True')
    
    import wandb
    
    
    wandb.login(key=args.online_logs)
    run = wandb.init(
    # Set the project where this run will be logged
    project=args.exp_name,
    # Track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "epochs": args.epoch,
    })
    
#-------------------------END---------------------






#train: training loop
def train(questions_onehot, train_dataloader, model, criterion, optimizer, device):
    #two loops
    #1- dataloader
    #2- loop for each question
    model.train(True)
    running_loss=0.0
    
    
    for data in train_dataloader:
        data= data.to(device)
        
        
        
        optimizer.zero_grad()
        
        input= data
        
        output= model(input)
        
        loss = criterion()
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
            
    return running_loss/(len(train_dataloader) * len(questions_onehot))
    


#eval: eval loop
def eval(questions_onehot, test_dataloader, model, criterion, device):
    
    model.eval()
    running_loss=0.0
    
    
    for data in test_dataloader:
        data= data.to(device)
        input= data
        
        output= model(input)
        
        
        loss = criterion()
        
        running_loss+=loss.item()
            
    return running_loss/(len(test_dataloader) * len(questions_onehot))

def prepare_model(args, device):

    #dataloader
    data= Dataset()
    
    
    test_size= int(len(data)*args.test_split)
    train_size = int(len(data) - test_size)
    
    train_dataset, test_dataset = random_split(data, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=args.shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch, shuffle=args.shuffle)
    

    
    #model
    model = BaseModel(#FILL) # audio length + length for one hot encoding
    model= model.to(device)
    
    #loss + optim
    criterion = #FILL
    optimizer= optim.Adam(model.parameters(), lr=args.lr)
    
    return train_dataloader, test_dataloader, questions_onehot, model, criterion, optimizer


#main: preparing model and dataloader
def main(args):
    art= """
     ^ ^           
    (O,O)          
    (   ) START    
    -"-"-----------
    """
    print(art)
    logging.info(art)
    
    #check device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ev_loss=float('inf')
    art2= """
  ___                        ___  
 (o o)                      (o o) 
(  V  ) Loading Components (  V  )
--m-m------------------------m-m--
          """

    print(art2)
    logging.info(art2)
    
    train_dataloader, test_dataloader, questions_onehot, model, criterion, optimizer = prepare_model(args, device)
    

    
    
    for i in tqdm(range(args.epoch)):
        
        training_loss= train(train_dataloader=train_dataloader, questions_onehot=questions_onehot, 
                                    model=model, criterion=criterion, optimizer=optimizer, device=device)
        
        eval_loss, sample= eval(test_dataloader=test_dataloader, questions_onehot=questions_onehot, 
                                model=model, criterion=criterion, device=device)
        #log stuff make log files
        if args.offline_logs:
            header1=f'-----------EPOCH {i+1}-----------'

            
            m1_1= f'training_loss: {training_loss}'
            m1_2= f'evaluation_loss: {eval_loss}'
            

            
            #print to console
            print(header1)
            print(m1_1)
            print(m1_2)

            
            #print to logs.log
            logging.info(header1)
            logging.info(m1_1)
            logging.info(m1_2)

        
        if args.online_logs:
            wandb.log({'training_loss': training_loss})
            wandb.log({'evaluation_loss': eval_loss})
            wandb.log({'epoch': i+1})
            
        #save model and things
        if eval_loss<ev_loss:
            ev_loss = eval_loss
            torch.save(model.state_dict(), os.path.join(args.save_path, args.exp_name,'best.pth'))
        
        
        
    



if __name__ == '__main__':
    
    main(args)
    logging.shutdown()
