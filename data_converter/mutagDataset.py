from dgl.data import DGLDataset
from processData import processData

class MUTAG(DGLDataset):
    def __init__(self):
        super().__init__(name='mutag')
        
    def process(self):
        getGraphData, _ = processData()
        graphData = getGraphData()
        self.graphs = []
        for graphID in graphData.keys():
            self.graphs.append(graphData[graphID])
           
    def __getitem__(self, i):
        return self.graphs[i]

    def __len__(self):
        return len(self.graphs)