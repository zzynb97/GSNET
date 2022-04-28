import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from models.GoogleNet.GoogleNet_Leaky import Net
import scipy.io as io
from cfgs import *

def gsnet(x, batch_size, epochs):
    # input x is a tensor type
    
    N = x.shape[0]
    input_np = x.detach().cpu().numpy().reshape(N, -1).astype(np.float32)
    net = Net(x)
    model = net.to(Device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=0.1)
    # omni-iPIM
    P = x2p(input_np)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.  # early exaggeration
    P = np.maximum(P, 1e-12)
    P = torch.from_numpy(P).to(Device).type(torch.double)
    
    # training
    for i in range(epochs):
        INDEX = np.random.permutation(N)
        for j in range(N // batch_size):
            data_input = x[INDEX[j * batch_size: (j+1) * batch_size], :, :, :]
            outputs_tensor = model(data_input)
            outputs_tensor = outputs_tensor.type(torch.double)
            n = outputs_tensor.shape[0]
            index_temp = np.ix_(INDEX[j * batch_size: (j+1) * batch_size], INDEX[j * batch_size: (j+1) * batch_size])
            P_temp = P[index_temp]
            P_temp = P_temp / torch.sum(P_temp)
            # oPIM
            Gram = torch.mm(outputs_tensor, outputs_tensor.t())
            diag_gram = torch.diag(Gram)
            H = diag_gram.repeat(n, 1)
            K = H.t()
            D = H + K - 2 * Gram
            D[range(n), range(n)] = 0.
            Q = 1. / (1. + D)
            Q = Q / torch.sum(Q)
            loss = torch.sum((P_temp - Q) * (torch.log(P_temp) - torch.log(Q)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (j + 1) % 10 == 0:
                print('epoch: %d/%d, %f' % (i + 1, epochs, loss.item())) 
                
    torch.save(model.state_dict(), './pts/GSNET_{}.pt'.format(args.dataset))        

def show_pattern(x, x_Dataset, category):
    print('------ eval process ------')
    net = Net(x)
    model = net.to(Device)
    model.load_state_dict(torch.load('./pts/GSNET_{}.pt'.format(category)))
    model.eval()
    data_eval = DataLoader(dataset=x_Dataset, batch_size=200, shuffle=False)
    if category == 'mnist':
        label = x_Dataset.targets.data.numpy()
    elif category == 'norb':
        label = x_Dataset.train_labels
    temp_data = torch.zeros((1, 2), dtype=torch.float32)
    for i, (data_input, data_label) in enumerate(data_eval):
        data_input = data_input.type(torch.float32).to(Device)
        outputs_batch = model(data_input)
        outputs_cpu = outputs_batch.cpu().detach()
        # print(outputs_batch.shape)
        temp_data = torch.vstack((temp_data, outputs_cpu))
    y1 = temp_data[1:, :]        
    y_np = y1.numpy()
    plot_embedding_2d(y_np, label, 'GSNET_{}'.format(args.dataset)) 
 
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GSNET',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'cifar10', 'norb'])
    # parser.add_argument('--number', default='all')
    parser.add_argument('--mode', default='eval',
                        choices=['train', 'eval'])
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--save_dir', default='results')
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # load data
    x_dataset, x = load_data(args.dataset)
    
    if args.mode == 'train':
        gsnet(x, args.batch_size, args.epochs)
        show_pattern(x, x_dataset, args.dataset)
        
    elif args.mode == 'eval':
        show_pattern(x, x_dataset, args.dataset)
        
        

    
