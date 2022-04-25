import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from models.GoogleNet.GoogleNet_Leaky import net
import scipy.io as io
from cfgs import Device, lr, LoadData, x2p, milestone, plot_embedding_2d

def gsnet(x, batch_size, epochs):
    N = x.shape[0]
    input_np = x.detach().cpu().numpy().reshape(N, -1).astype(np.float32)
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
                
    # show pattern
    model.eval()
    data_eval = DataLoader(dataset=data_load, batch_size=20, shuffle=True)
    temp_data = torch.zeros((1, 2), dtype=torch.float32)
    temp_label = torch.zeros((1, 1))
    for i, (data_input, data_label) in enumerate(data_eval):
        data_input = data_input.type(torch.float32).view(data_input.shape[0], 1, 28, 28).to(Device)
        outputs_batch = model(data_input)
        outputs_cpu = outputs_batch.cpu().detach()
        # print(outputs_batch.shape)
        temp_data = torch.vstack((temp_data, outputs_cpu))
        temp_label = torch.vstack((temp_label, data_label.view(-1, 1)))
    y1 = temp_data[1:, :]        
    y_np = y1.numpy()
    label = temp_label[1:, :]
    label = label.numpy()
    plot_embedding_2d(y_np, label, 'GSNET_MNIST')
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='GSNET',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='mnist',
                        choices=['mnist', 'cifar10', 'norb'])
    parser.add_argument('--number', default='all')
    parser.add_argument('--mode', default='train',
                        choices=['train', 'eval'])
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--epochs', default=1000, type=int)
    parser.add_argument('--save_dir', default='results')
    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # load data
    data_load = LoadData(mode=args.mode, data=args.dataset, number=int(args.number))
    input_np = data_load.train_data.astype(np.float32)
    if args.dataset == 'mnist':
        input_tensor = torch.from_numpy(input_np).to(Device).type(torch.float32)\
                                    .view(input_np.shape[0], 1, 28, 28)
    
    if args.mode == 'train':
        gsnet(input_tensor, args.batch_size, args.epochs)

    
