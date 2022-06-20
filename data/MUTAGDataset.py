import dgl
from dgl.data import DGLDataset
import torch
from createGraphData import * 

class MUTAG(DGLDataset):
    def __init__(self):
        super().__init__(name='mutag')
        
    def process(self):
        '''
        Graph Label:

        -1 nonmutagenic
        1  mutagenic
        '''
        graphDict, graphLabels = createMutagGraphs()
        self.graphs = []
        for graphID in graphDict.keys():
            graphdata = graphDict[graphID]
            self.graphs.append([dgl.heterograph(graphdata), torch.tensor(graphLabels[graphID - 1])])
            
    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs) 

class MUTAGNode(DGLDataset):
    def __init__(self):
        super().__init__(name='mutag')
        
    def process(self):
        '''
        Graph Label:

        -1 nonmutagenicxs
        1  mutagenic
        '''
        graphData, nodeLabels = createMUTAGNodes()
        self.graph = dgl.heterograph(graphData)
    
        
        n_nodes = len(nodeLabels)
        n_train = int(n_nodes * 0.8)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        test_mask[n_train:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return len(self.graph) 
