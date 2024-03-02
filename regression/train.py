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
from torch_geometric.nn import GAT, MLP, GCNConv,GATConv, global_mean_pool, global_add_pool
from networkx.classes.function import neighbors
from torch_geometric.data import Dataset, Data
import torch.nn.functional as F
import pickle
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

  num_classes= df_train[graph_labels_idx][0].nunique()

  # valid data

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

  print('Create class GNN')
  class GNN(torch.nn.Module):
      def __init__(self, num_node_features, hidden_channels, num_edge_features):
          super(GNN, self).__init__()
          torch.manual_seed(12345)
          self.conv1 = GATConv(num_node_features, hidden_channels, edge_dim=num_edge_features)
          self.conv2 = GATConv(hidden_channels, hidden_channels, edge_dim=num_edge_features)
          self.lin = Linear(hidden_channels, 1)

      def forward(self, x, edge_index, edge_attr, batch=None):
          # Node embedding
          x = self.conv1(x, edge_index, edge_attr=edge_attr)
          x = x.relu()
          x = self.conv2(x, edge_index, edge_attr=edge_attr) 
          # Readout layer
          batch = torch.zeros(x.shape[0], dtype=int) if batch is None else batch
          x = global_add_pool(x, batch)
          # Final classifier
          x = F.dropout(x, p=0.5, training=self.training)
          x = self.lin(x)
          # x = F.softmax(x, dim=1)

          return x  # Add this line to return the final output

  print("Model Created")
  model = GNN(data_list[0].num_features, 16, data_list[0].edge_attr.shape[1])
  print(model)

  learning_rate = 0.0001
  print("l_rate ",learning_rate)
  decay = 5e-4
  optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=decay)

  criterion = torch.nn.MSELoss()

  def train():
      total_loss=0
      i=0
      # print(len(data_list))
      for data in data_list:
        # print(i,":", end="")
        i+=1
        # print(i)
        model.train()
        optimizer.zero_grad()
        # Use all data as input, because all nodes have node features
        out = model.forward(data.x, data.edge_index, data.edge_attr)
        # var=torch.zeros(num_classes)
        # var[int(data.y)]=1
        # print(out[0],var)
        # Only use nodes with labels available for loss calculation --> mask
        loss = criterion(out[0],data.y)
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
    # if epoch % 100 == 0:
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
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

  plt.plot(range(1,51),train_losses,label="Training Loss")
  plt.plot(range(1,51),val_losses,label="Validation Loss")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()
  plt.savefig("reg_loss.png")

  # print('Start Testing')

  model_file = "model_reg_final.pkl"
  with open(model_file, 'wb') as file:
      pickle.dump(model, file)

else:
  model_file = "model_reg_final.pkl"
  with open(model_file, 'rb') as file:
      model_new = pickle.load(file)

    # print("Print model's weight")
    # for name, param in model_new.named_parameters():
    #     if param.requires_grad:
    #         print(name, param.data)

  loss=0
  for data in data_list:
    pred=model_new.forward(data.x, data.edge_index, data.edge_attr)
    # pred=pred.(dim=1)
    loss+=(float(pred)-float(data.y))**2
    print(f"{float(pred):.4f} {float(data.y):.4f} {(float(pred)-float(data.y)):.4f}")
  loss/=len(data_list)
  print(f"Loss: {loss:.4f}")