import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn
from torch.nn import Linear
from torch_geometric.data import DataLoader
import gzip
import os
from io import StringIO
from torch_geometric.utils import to_networkx
from torch_geometric.loader import NeighborLoader as nl
import networkx as nx
from torch_geometric.nn import GAT, MLP, GCNConv, GATConv, global_mean_pool, global_add_pool
from networkx.classes.function import neighbors
from torch_geometric.data import Dataset, Data
import torch.nn.functional as F
import pickle
from torch_geometric.data.dataset import to_list
from sklearn.metrics import roc_auc_score
import evaluate
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Training a classification model")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--val_dataset_path", required=True)
    args = parser.parse_args()
    print(f"Training a classification model. Output will be saved at {args.model_path}. Dataset will be loaded from {args.dataset_path}. Validation dataset will be loaded from {args.val_dataset_path}.")

if __name__=="__main__":
    main()

# from torch_geometric.nn import MLP
# from torch import tensor as tt

data_list=[]
data_list_val=[]

print(len(sys.argv))
if len(sys.argv) >= 6:
  files=["train"]

  train_files=[]
  df_train=[]
  print('Load dataset')
  # for item in files:
  #   path=os.path.join(dir,item)
  path = sys.argv[4]
  for graph_file in os.listdir(path):
    train_files.append(graph_file)
    with gzip.open(os.path.join(path,graph_file), "rb") as gz_file:
      content=gz_file.read().decode("utf-8")
      content=StringIO(content)
      x=pd.read_csv(content, header=None)
      df_train.append(x)
  print('Dataset loaded')
  num_nodes_idx = train_files.index("num_nodes.csv.gz")
  node_features_idx = train_files.index("node_features.csv.gz")
  num_edges_idx = train_files.index("num_edges.csv.gz")
  edge_featuress_idx = train_files.index("edge_features.csv.gz")
  graph_labels_idx = train_files.index("graph_labels.csv.gz")
  edges_idx = train_files.index("edges.csv.gz")

  null_idx=df_train[graph_labels_idx].isnull()
  # null_idx

  num_graphs=df_train[num_nodes_idx].shape[0]
  node_ptr=0
  edge_ptr=0
  # node_features_concat=[]
  # edge_features_concat=[]
  # edge_concat=[]
  print("Creating data_list")
  for x in range(num_graphs):

    # if x==2:
    #   break
    num_nodes=df_train[num_nodes_idx][0][x]
    node_features=torch.tensor(df_train[node_features_idx].loc[node_ptr:node_ptr+num_nodes-1].values, dtype=torch.float32)
    # node_features_concat.append(node_features)

    num_edges=df_train[num_edges_idx][0][x]
    edges=torch.tensor(df_train[edges_idx].loc[edge_ptr:edge_ptr+num_edges-1].values, dtype=torch.float32)
    edges=np.transpose(edges)
    # edge_concat.append(edges)

    edge_features=torch.tensor(df_train[edge_featuress_idx].loc[edge_ptr:edge_ptr+num_edges-1].values, dtype=torch.float32)
    # edge_features_concat.append(edge_features)

    graph_label=torch.tensor(df_train[graph_labels_idx][0][x], dtype=torch.float32)

    node_ptr=node_ptr+num_nodes
    edge_ptr=edge_ptr+num_edges

    if null_idx[0][x]==True:
      continue
    data=Data(x=node_features, edge_index=edges.to(torch.int64), edge_attr=edge_features, y=graph_label)
    data_list.append(data)

    # print(data.edge_index)
    # print(data)
    
  print("data_list created")

  # elif sys.argv[1] == 'valid'::
  # files=["valid"]

  train_files=[]
  df_train=[]
  print('Load dataset')
  # for item in files:
  #   path=os.path.join(dir,item)
  path = sys.argv[6]
  for graph_file in os.listdir(path):
    train_files.append(graph_file)
    with gzip.open(os.path.join(path,graph_file), "rb") as gz_file:
      content=gz_file.read().decode("utf-8")
      content=StringIO(content)
      x=pd.read_csv(content, header=None)
      df_train.append(x)
  print('Dataset loaded')
  num_nodes_idx = train_files.index("num_nodes.csv.gz")
  node_features_idx = train_files.index("node_features.csv.gz")
  num_edges_idx = train_files.index("num_edges.csv.gz")
  edge_featuress_idx = train_files.index("edge_features.csv.gz")
  graph_labels_idx = train_files.index("graph_labels.csv.gz")
  edges_idx = train_files.index("edges.csv.gz")

  null_idx=df_train[graph_labels_idx].isnull()
  # null_idx

  num_graphs=df_train[num_nodes_idx].shape[0]
  node_ptr=0
  edge_ptr=0
  # node_features_concat=[]
  # edge_features_concat=[]
  # edge_concat=[]
  print("Creating data_list_val")
  for x in range(num_graphs):

    # if x==2:
    #   break
    num_nodes=df_train[num_nodes_idx][0][x]
    node_features=torch.tensor(df_train[node_features_idx].loc[node_ptr:node_ptr+num_nodes-1].values, dtype=torch.float32)
    # node_features_concat.append(node_features)

    num_edges=df_train[num_edges_idx][0][x]
    edges=torch.tensor(df_train[edges_idx].loc[edge_ptr:edge_ptr+num_edges-1].values, dtype=torch.float32)
    edges=np.transpose(edges)
    # edge_concat.append(edges)

    edge_features=torch.tensor(df_train[edge_featuress_idx].loc[edge_ptr:edge_ptr+num_edges-1].values, dtype=torch.float32)
    # edge_features_concat.append(edge_features)

    graph_label=torch.tensor(df_train[graph_labels_idx][0][x], dtype=torch.float32)

    node_ptr=node_ptr+num_nodes
    edge_ptr=edge_ptr+num_edges

    if null_idx[0][x]==True:
      continue
    data=Data(x=node_features, edge_index=edges.to(torch.int64), edge_attr=edge_features, y=graph_label)
    data_list_val.append(data)

    # print(data.edge_index)
    # print(data)
  print("data_list_val created")

  num_classes= df_train[graph_labels_idx][0].nunique()

  print('Create class GNN')
  class GNN(torch.nn.Module):
      def __init__(self, num_node_features, hidden_channels, num_edge_features):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(num_node_features, hidden_channels, edge_dim=num_edge_features)
        self.conv2 = GATConv(hidden_channels, hidden_channels, edge_dim=num_edge_features)
        self.lin = Linear(hidden_channels, 2)

      def forward(self, x, edge_index, edge_attr, batch=None):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        batch = torch.zeros(x.shape[0], dtype=int) if batch is None else batch
        x = global_add_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x 

  print("Model Created")
  model = GNN(data_list[0].num_features, 16, data_list[0].edge_attr.shape[1])
  # print(model)
  learning_rate = 0.0001
  print("l_rate ",learning_rate)
  decay = 5e-4
  optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=decay)
  criterion = torch.nn.CrossEntropyLoss()

  def train():
      total_loss=0
      # print(len(data_list))
      for data in data_list:
        model.train()
        optimizer.zero_grad()
        out = model.forward(data.x, data.edge_index, data.edge_attr)
        var=torch.zeros(num_classes)
        var[int(data.y)]=1
        # print(out[0],var)
        # Only use nodes with labels available for loss calculation --> mask
        loss = criterion(out[0],var)
        # print(loss)
        total_loss+=loss
        loss.backward()
        optimizer.step()
      return total_loss/len(data_list)

  print('Train function created')

  # def test():
  #       model.eval()
  #       out = model(data.x, data.edge_index)
  #       # Use the class with highest probability.
  #       pred = out.argmax(dim=1)
  #       # Check against ground-truth labels.
  #       test_correct = pred == data.y
  #       # Derive ratio of correct predictions.
  #       test_acc = int(test_correct.sum()) / int(data.sum())
  #       return test_acc

  print('Start Training')
  train_losses = []
  val_losses = []
  for epoch in range(0, 50):
    train_loss = train()
    train_losses.append(train_loss.detach().item())
    val_loss = 0
    for data in data_list_val:
      out = model.forward(data.x, data.edge_index, data.edge_attr)
      var=torch.zeros(num_classes)
      var[int(data.y)]=1
      # print(out[0],var)
      # Only use nodes with labels available for loss calculation --> mask
      loss = criterion(out[0],var)
      # print(loss)
      val_loss+=loss
    val_loss /= len(data_list_val)
    val_losses.append(val_loss.detach().item())

    if epoch % 100 == 0:
      print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

  plt.plot(range(1,51),train_losses,label="Training Loss")
  plt.plot(range(1,51),val_losses,label="Validation Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig("class_loss.png")
  model_file = sys.argv[2]
  with open(model_file, 'wb') as file:
      pickle.dump(model, file)
  

# print('Start Testing')
else:
  print('Create class GNN')
  class GNN(torch.nn.Module):
      def __init__(self, num_node_features, hidden_channels, num_edge_features):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GATConv(num_node_features, hidden_channels, edge_dim=num_edge_features)
        self.conv2 = GATConv(hidden_channels, hidden_channels, edge_dim=num_edge_features)
        self.lin = Linear(hidden_channels, 2)

      def forward(self, x, edge_index, edge_attr, batch=None):
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        batch = torch.zeros(x.shape[0], dtype=int) if batch is None else batch
        x = global_add_pool(x, batch)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x 
  model_file = sys.argv[2]
  with open(model_file, 'rb') as file:
      model_new = pickle.load(file)

  # print("Print model's weight")
  # for name, param in model_new.named_parameters():
  #     if param.requires_grad:
  #         print(name, param.data)

  count=0
  pred_label=[]
  true_label=[]
  prob_soft=[]

  for data in data_list:
    pred=model_new.forward(data.x, data.edge_index, data.edge_attr)
    true_l=int((data.y).numpy())
    pred_l=int((pred.argmax(dim=1)).numpy())
    out=(F.softmax(pred,dim=1)).detach().numpy()
    # soft_0=out[0][0]
    soft_1=out[0][1]
    # print(true_l)
    # print(pred_l)
    # print(soft_0)
    # print(soft_1)
    # print()
    # print(pred_arg[0])
    # print()
    # print(pred_arg)
    # print(pred_arg[0])
    # print("pred: ", pred)
    # value, index = torch.max(pred, dim=0)
    # print(max(value.detach().numpy()))
    # print(index)
    # print("Maximum value:", value.item() if value.numel() == 1 else value.tolist())
    # print("Index of maximum value:", index.item() if index.numel() == 1 else index.tolist())
    # print()
    # print("pred: ", pred[0])

    # print(pred)
    # print(pred_arg[0])
    # print(pred)
    # pred=pred.detach().numpy()
    # print(pred[0])
    # print()
    # print(out)
    true_label.append(true_l)
    pred_label.append(pred_l)
    prob_soft.append(soft_1)
    # print(out)
    # print(out)
    # pred_label.append(pred_arg[0])
    # lab= int((data.y).numpy())
    # # print(lab)
    # lab=
    # true_label.append(lab)

    # val=
    # print("pred_arg: ",[pred_arg.numpy()][0])
    # print("pred_arg_prob: ", pred[pred_arg])
    # print(int(pred), int(data.y))
    # if(int(pred) == int(data.y)):
    #   count+=1

  # print(f"Accuracy: {(count/len(data_list)*100):.4f}%")
  # print(true_label[1])
  # print(pred_label[1])
  # print("*")

  prob_soft=np.array(prob_soft)
  true_label=np.array(true_label)
  # evaluate.tocsv(prob_soft,task="classification")
  # obj=evaluate.Evaluator()
  # score=obj._eval_rocauc(true_label,prob_soft)
  # print("Score: ",score)
  print("roc_auc_score:", roc_auc_score(true_label,prob_soft))


