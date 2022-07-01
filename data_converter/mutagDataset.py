from dgl.data import DGLDataset
import dgl
import torch
from processData import processData
from processDataOneNtype import processDataOneNtype

class MUTAG(DGLDataset):
    def __init__(self):
        super().__init__(name='mutag')
        
    def process(self):
        getGraphData, _ = processData()
        graphData = getGraphData()
        self.graphs = []
        for graphID in graphData.keys():
            graph, label = graphData[graphID]
            self.graphs.append([dgl.heterograph(graph), label])
           
    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)


class MUTAGOneNtype(DGLDataset):
    def __init__(self):
        super().__init__(name='mutag')
        
    def process(self):
        graphData = processDataOneNtype()
        self.graphs = []
        for graphID in graphData.keys():
            graph, label = graphData[graphID]
            self.graphs.append([dgl.heterograph(graph), label])
           
    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)