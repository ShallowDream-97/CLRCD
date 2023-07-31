import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, ValTestDataLoader
from model import Net
from utils import CommonArgParser, construct_local_map
from GraphEva import GraphEva

def train(args, local_map):
    data_loader = TrainDataLoader()
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    
    grapheva = GraphEva(args, local_map)
    grapheva = grapheva.to(device)
    net = Net(args, local_map)
    net = net.to(device)
    net.load_state_dict(torch.load('model/model_epoch1_dropout:0.05_ssl_para:1e-05_reg_para:-1e-07'))
    grapheva.load_state_dict(torch.load('model/graph_model_epoch1_dropout:0.05_ssl_para:1e-05_reg_para:-1e-07'))

    rmse, auc = predict(args, grapheva,net, 1)


def predict(args, grapheva,net, epoch):
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    data_loader = ValTestDataLoader('predict')
    print('predicting model...')
    data_loader.reset()
    net.eval()
    grapheva.eval()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        all_stu_emb_ssl,ssl_loss_train = grapheva.forward(input_stu_ids)
        output = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs,all_stu_emb_ssl)
        output = output.view(-1)
        # compute accuracy
        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    # compute accuracy
    accuracy = correct_count / exer_count
    # compute RMSE
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    # compute AUC
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    with open('result/ncd_model_val_'+str(epoch + 1)+ 'dropout:'+str(args.dropout)+'_ssl_para:'+str(args.ssl_para)+'_reg_para:'+str(args.reg_para)+'.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

    return rmse, auc

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    train(args, construct_local_map(args))
