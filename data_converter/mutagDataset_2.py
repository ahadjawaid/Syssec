from dgl.data import DGLDataset
import dgl
import torch
from processData_2 import processData

class MUTAG(DGLDataset):
    def __init__(self):
        super().__init__(name='mutag')
        
    def process(self):
        graphData = processData()
        self.graphs = []
        for graphID in graphData.keys():
            graph, label = graphData[graphID]
            self.graphs.append([dgl.heterograph(graph), label])
           
    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)
