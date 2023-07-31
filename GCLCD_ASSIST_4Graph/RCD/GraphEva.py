import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np
from GraphLayer import GraphLayer
import dgl.nn as dglnn

class AttentionFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionFusion, self).__init__()
        self.linear_layers = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(4)])
        self.attention = nn.Linear(output_dim, 1)
    
    def forward(self, graph_outputs):
        h = [F.relu(linear(x)) for linear, x in zip(self.linear_layers, graph_outputs)]
        h = torch.stack(h, dim=0)  # Shape: [4, batch_size, output_dim]
        
        attention_weights = F.softmax(self.attention(h), dim=0)
        
        fused_output = torch.sum(attention_weights * h, dim=0)  # Shape: [batch_size, output_dim]
        
        return fused_output
    
    
class GATLayer(nn.Module):
    def __init__(self, g,in_dim, out_dim, num_heads):
        super(GATLayer, self).__init__()
        g = dgl.add_self_loop(g)
        self.g = g
        self.num_heads = num_heads
        self.gat = dglnn.GATConv(in_dim, out_dim, num_heads)
        self.fc = nn.Linear(num_heads * out_dim, out_dim)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        
        
    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': a}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        h = self.gat(self.g, h).view(-1, self.num_heads * h.size(-1))
        z = self.fc(h)
        self.g.ndata['z'] = z
        self.g.apply_edges(self.edge_attention)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
       
    
def remove_random_edges(g, fraction=0.1):
    # 获取图的边列表
    edges = g.edges()
    
    # 计算需要删除的边的数量
    num_edges = len(edges[0])
    num_remove = int(num_edges * fraction)
    
    # 随机选择一部分边进行删除
    remove_indices = np.random.choice(num_edges, num_remove, replace=False)    
    # 创建一个新的图，不包含被删除的边
    new_g = dgl.remove_edges(g, remove_indices)
    
    return new_g

class GraphEva(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim        
        
        #self.e_from_u = local_map['e_from_u'].to(self.device)

        super(GraphEva, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.knowledge_dim)

        self.stu_index = torch.LongTensor(list(range(self.emb_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_n))).to(self.device)
        #Graph
        self.u_from_e_correct = local_map['u_from_e_correct'].to(self.device)
        self.u_from_e_incorrect = local_map['u_from_e_incorrect'].to(self.device)
        self.e_from_u_correct = local_map['e_from_u_correct'].to(self.device)
        self.e_from_u_incorrect = local_map['e_from_u_incorrect'].to(self.device)
        
        self.u_from_e_incorrect_per = remove_random_edges(local_map['u_from_e_incorrect']).to(self.device)
        
        
        # self.u_from_e_correct_per = local_map['u_from_e_correct_per'].to(self.device)
        # self.u_from_e_incorrect_per = local_map['u_from_e_incorrect_per'].to(self.device)
        # self.e_from_u_correct_per = local_map['e_from_u_correct_per'].to(self.device)
        # self.e_from_u_incorrect_per = local_map['e_from_u_incorrect_per'].to(self.device)
        
        self.u_from_e_correct_GAT = GATLayer(self.u_from_e_correct,args.knowledge_n, args.knowledge_n, num_heads=3)
        self.u_from_e_incorrect_GAT = GATLayer(self.u_from_e_incorrect,args.knowledge_n, args.knowledge_n, num_heads=3)
        self.e_from_u_correct_GAT = GATLayer(self.e_from_u_correct,args.knowledge_n, args.knowledge_n, num_heads=3)
        self.e_from_u_incorrect_GAT = GATLayer(self.e_from_u_incorrect,args.knowledge_n, args.knowledge_n, num_heads=3)
        #对哪里扰动影响最大
        #self.u_from_e_correct_GAT_per = GATLayer(self.u_from_e_correct_per,args.knowledge_n, args.knowledge_n, num_heads=3)
        self.u_from_e_incorrect_GAT_per = GATLayer(self.u_from_e_incorrect_per,args.knowledge_n, args.knowledge_n, num_heads=3)
        
        #self.e_from_u = GATLayer(self.e_from_u,args.knowledge_n, args.knowledge_n, num_heads=3)
        #self.u_from_e_GAT_per = GATLayer(self.u_from_e_2, args.knowledge_n, args.knowledge_n, num_heads=3)
        self.ExpressionLayer1 = ExpressionDooubleEva(args, self.u_from_e_correct_GAT,self.u_from_e_incorrect_GAT,self.e_from_u_correct_GAT,self.e_from_u_incorrect_GAT,self.u_from_e_correct_GAT,self.u_from_e_incorrect_GAT_per)
        self.ExpressionLayer2 = ExpressionSingleEva(args, self.u_from_e_correct_GAT,self.u_from_e_incorrect_GAT)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        
        # Fusion layer 1
        all_stu_emb1,all_stu_emb_per = self.ExpressionLayer1(exer_emb, all_stu_emb)
        # Fusion layer 2
        all_stu_emb2 = self.ExpressionLayer2(exer_emb, all_stu_emb1)
        all_stu_emb2_per = self.ExpressionLayer2(exer_emb, all_stu_emb_per)

        # get batch student data
        batch_stu_emb = all_stu_emb2[stu_id] # 32 123
       
        # get batch student data PER!!!
        batch_stu_emb_per = all_stu_emb2_per[stu_id] # 32 123
        # 归一化为单位向量
        batch_stu_emb_normalized = torch.nn.functional.normalize(batch_stu_emb, p=2, dim=1)
        batch_stu_emb_1_normalized = torch.nn.functional.normalize(batch_stu_emb_per, p=2, dim=1)

        # 计算余弦相似度
        similarity = torch.matmul(batch_stu_emb_normalized, batch_stu_emb_1_normalized.T)
        soft_para = 0.2
        similarity = similarity/soft_para
        # 应用 exp 函数
        exp_similarity = torch.exp(similarity)
        # 计算相同索引行的余弦相似度
        diag_similarities = exp_similarity.diag()
        # 初始化总损失
        ssl_loss = 0.0
        # print("Compute Sim in the batch:")
        for u in (range(batch_stu_emb.shape[0])):
             # 获取当前行的余弦相似度
            u_row_similarity = similarity[u, :]
            u_u_similarity = diag_similarities[u]
            # 计算与当前行的余弦相似度之和
            sum_similarity = torch.sum(u_row_similarity)

            # 除以余弦相似度之和
            divided_similarity = u_u_similarity / (sum_similarity + 1e-8)
            # 计算当前行的损失
            row_loss = torch.sum(divided_similarity)
            # print(type(row_loss))
            row_loss =-torch.log(torch.clamp(row_loss, min=1e-8))
            # 累积到总损失
            ssl_loss += row_loss
            # print(time.time()-similarity_com_bef)
            # print("ssl_loss:"+str(ssl_loss))
        #ssl_loss = ssl_loss/batch_stu_emb.shape[0]
        return all_stu_emb2,ssl_loss


class ExpressionDooubleEva(nn.Module):
    def __init__(self, args, u_from_e_correct,u_from_e_incorrect ,e_from_u_correct,e_from_u_incorrect,u_from_e_correct_per, u_from_e_incorrect_per):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim

        # graph structure

        super(ExpressionDooubleEva, self).__init__()
        
        self.u_from_e_correct = u_from_e_correct
        self.u_from_e_incorrect = u_from_e_incorrect
        self.e_from_u_correct = e_from_u_correct
        self.e_from_u_incorrect = e_from_u_incorrect
        
        #self.u_from_e_correct_per = u_from_e_correct_per
        self.u_from_e_incorrect_per = u_from_e_incorrect_per
        
        self.fusion_layer = nn.Linear(4 * self.knowledge_dim, self.knowledge_dim)



    def forward(self, exer_emb, all_stu_emb):
        e_u_graph = torch.cat((exer_emb, all_stu_emb), dim=0)
        u_from_e_correct_graph = self.u_from_e_correct(e_u_graph)
        u_from_e_incorrect_graph = self.u_from_e_incorrect(e_u_graph)
        
        #u_from_e_correct_graph_per = self.u_from_e_correct_per(e_u_graph)
        u_from_e_incorrect_graph_per = self.u_from_e_incorrect_per(e_u_graph)
        
        e_from_u_correct_graph = self.e_from_u_correct(e_u_graph)
        e_from_u_incorrect_graph = self.e_from_u_incorrect(e_u_graph)
        
        all_graphs = torch.cat((u_from_e_correct_graph, u_from_e_incorrect_graph, e_from_u_correct_graph, e_from_u_incorrect_graph), dim=-1)
        all_graphs_per = torch.cat((u_from_e_correct_graph, u_from_e_incorrect_graph_per, e_from_u_correct_graph, e_from_u_incorrect_graph), dim=-1)
        
        
        fusion_output = self.fusion_layer(all_graphs)
        fusion_output_per = self.fusion_layer(all_graphs_per)
        
        all_stu_emb_0 = all_stu_emb
        all_stu_emb = all_stu_emb + fusion_output[self.exer_n:]
        all_stu_emb_per = all_stu_emb_0 + fusion_output_per[self.exer_n:]
        
        
        # u_from_e_graph_per = self.u_from_e_per(e_u_graph)

        # # updated students
        # all_stu_emb_0 = all_stu_emb
        # all_stu_emb = all_stu_emb + u_from_e_graph[self.exer_n:]
        # all_stu_emb_per = all_stu_emb_0 + u_from_e_graph_per[self.exer_n:]

        return all_stu_emb, all_stu_emb_per
    
class ExpressionSingleEva(nn.Module):
    def __init__(self, args, u_from_e_correct, u_from_e_incorrect):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim

        # graph structure

        super(ExpressionSingleEva, self).__init__()
        self.u_from_e_correct = u_from_e_correct
        self.u_from_e_incorrect = u_from_e_incorrect
        self.fusion_layer = nn.Linear(2 * self.knowledge_dim, self.knowledge_dim)


    def forward(self, exer_emb, all_stu_emb):
        e_u_graph = torch.cat((exer_emb, all_stu_emb), dim=0)
        u_from_e_correct_graph = self.u_from_e_correct(e_u_graph)
        u_from_e_incorrect_graph = self.u_from_e_incorrect(e_u_graph)
        
        all_graphs = torch.cat((u_from_e_correct_graph, u_from_e_incorrect_graph), dim=-1)
        fusion_output = self.fusion_layer(all_graphs)
        all_stu_emb = all_stu_emb + fusion_output[self.exer_n:]


        return all_stu_emb