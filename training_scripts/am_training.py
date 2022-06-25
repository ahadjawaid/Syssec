import dgl
import dgl.function as fn
import torch
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import AMDataset

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv4 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, feat, eweight=None):
        # inputs are features of nodes
        with graph.local_scope():
            feat = self.conv1(graph, feat)
            feat = {k: F.relu(v) for k, v in feat.items()}
            feat = self.conv2(graph, feat)
            feat = {k: F.relu(v) for k, v in feat.items()}
            feat = self.conv3(graph, feat)
            feat = {k: F.relu(v) for k, v in feat.items()}
            feat = self.conv4(graph, feat)
            graph.ndata['h'] = feat
            if eweight is None:
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            else:
                graph.edata['w'] = eweight
                graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
            return graph.ndata['h']

def train(model, hetero_graph, node_features, epochs, printInterval):
    opt = torch.optim.Adam(model.parameters())
    train_mask = g.nodes[category].data['train_mask']
    test_mask = g.nodes[category].data['test_mask']
    labels = g.nodes[category].data['labels']

    for epoch in range(epochs):
        model.train()
        # forward propagation by using all nodes and extracting the user embeddings
        logits = model(hetero_graph, node_features)[category]
        pred = logits.argmax(1)
        # compute loss
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        # Compute validation accuracy.  Omitted in this example.
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if epoch % printInterval == 0:
            print('In epoch {}, loss: {:.3f}, train acc: {:.3f}, test acc: {:.3f})'.format(
                epoch, loss,train_acc, test_acc))
    print('In epoch {}, loss: {:.3f}, train acc: {:.3f}, test acc: {:.3f})'.format(
                epoch, loss,train_acc, test_acc))

device = torch.device("cuda:0")

dataset = AMDataset()
g = dataset[0].to(device)

num_classes = dataset.num_classes
category = dataset.predict_category

features = {}
for ntype in g.ntypes:
    features[ntype] = torch.zeros((g.num_nodes(ntype), 10)).to(device)

model = RGCN(10, 20, num_classes, g.etypes).to(device)
train(model, g, features, epochs=100, printInterval=10)
torch.save(model, 'AM_Trained_Model.pt')
