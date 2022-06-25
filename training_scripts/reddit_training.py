import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GraphConv

class Model(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_feats=20):
        '''
        in_feats: Input features
        out_feats: Output features
        hidden_feats: Hidden layer features
        '''
        super(Model, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        self.conv2 = GraphConv(hidden_feats, out_feats)
        
    def forward(self, graph, feat, eweight=None):
        with graph.local_scope():
            feat = self.conv1(graph, feat)
            feat = F.relu(feat)
            feat = self.conv2(graph, feat)
            graph.ndata['h'] = feat
            if eweight is None:
                graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            else:
                graph.edata['w'] = eweight
                graph.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h'))
            return graph.ndata['h']

def train(model, g, epochs=10, printInterval=5, lr=0.001):
    '''
    model: Training Model
    g: Training graph
    epochs: Number of epochs
    printInterval: Interval that data is displayed at
    lr: Learning rate
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    
    for epoch in range(epochs):
        logits = model(g, features)
        pred = logits.argmax(1)
        
        loss = criterion(logits[train_mask], labels[train_mask])
        
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print('In epoch {}, loss: {:.3f}, val acc: {:.3f}, test acc: {:.3f})'.format(epoch, loss, val_acc, test_acc))
    
data = RedditDataset()
g = data[0]

features = g.ndata['feat']
model = Model(features.shape[1], data.num_classes)
train(model, g, epochs=2, printInterval=1)
torch.save(model, 'Reddit_Trained_Model.pt')