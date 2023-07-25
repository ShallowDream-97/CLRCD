import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import json
import sys
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
from utils import CommonArgParser, construct_local_map

def train(args, local_map):
    # rank = args.rank

    print(torch.cuda.is_available())

    dist.init_process_group(backend='nccl')
    rank = dist.rank()
    device = torch.device(('cuda:%d' % (rank)))
    print(device)
    torch.cuda.set_device(device)
    

    data_loader = TrainDataLoader()
    data_loader = DistributedSampler(data_loader)

    net = Net(args, local_map)
    net = net.to(device)
    net = DistributedDataParallel(net, device_ids=[rank])

    print (net)
    f=open('res.txt','w')
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    print('training model...')

    loss_function = nn.NLLLoss()
    for epoch in range(args.epoch_n):
        data_loader.reset()
        running_loss = 0.0
        batch_count = 0
        while not data_loader.is_end():
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output+1e-10), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                f.write("loss:"+str(epoch + 1)+":"+str(running_loss / 200))
                running_loss = 0.0

        # test and save current model every epoch
        save_snapshot(net, 'model/model_epoch' + str(epoch + 1))
        rmse, auc = predict(args, net, epoch)
        f.write("RMSE:"+str(rmse)+"  AUC:"+str(auc))

# 修改其他的函数类似地反映这些改变，如用 `DistributedSampler` 和适应多设备。

if __name__ == '__main__':
    parser = CommonArgParser()
    #parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()
    train(args, construct_local_map(args))
