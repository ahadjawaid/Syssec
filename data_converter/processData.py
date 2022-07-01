import os
import torch

def processData():
    localPaths = ['MUTAG/MUTAG_A.txt', 
                   'MUTAG/MUTAG_graph_indicator.txt',
                   'MUTAG/MUTAG_node_labels.txt', 
                   'MUTAG/MUTAG_graph_labels.txt',
                   'MUTAG/MUTAG_edge_labels.txt']

    globalPath = []
    for path in localPaths:
        globalPath.append(joinDirPathToLocalPath(path))

    with open(globalPath[0], 'r') as edgeRelFile, \
         open(globalPath[1], 'r') as graphIndicatorFile, \
         open(globalPath[2], 'r') as nodeTypeFile, \
         open(globalPath[3], 'r') as graphLabelFile, \
         open(globalPath[4], 'r') as edgeTypeFile:

        elements = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
        ntypeKeys = encodeToKey(elements)
        ntypeDict = createKeyValueDict(ntypeKeys, nodeTypeFile)

        bonds = ['aromatic', 'single', 'doub', 'triple']
        etypeKey = encodeToKey(bonds)
        edgeDict = createEdgeDict(ntypeDict, etypeKey, edgeTypeFile, edgeRelFile)

        graphIndicator = fileToList(graphIndicatorFile, dtype=int)
        graphKey = encodeToKey(graphIndicator, start=1, dtype=int)
        graphDict = createGraphDict(graphKey, edgeDict)
        graphLabels = fileToList(graphLabelFile, dtype=int)
        graphData = createLabeledGraphDict(graphDict, graphLabels)

        nodeLabels = createNodeLabelsDict(graphLabels, graphIndicator, ntypeDict)

    def getGraphData():
        return graphData


    def getNodeData():
        return edgeDict, nodeLabels

    return getGraphData, getNodeData
        


def joinDirPathToLocalPath(local_path):
    return os.path.join(os.path.dirname(__file__), local_path)


def encodeToKey(keys, start=0, dtype=str):
    return {encoding: dtype(key) for encoding, key in enumerate(keys, start)}


def fileToList(file, dtype=int):
    return [dtype(elem) for elem in file]


def createKeyValueDict(key, labels):
    dict = {}
    for id, label in enumerate(labels, 1):
        dict[id] = key[int(label)]

    return dict


def createEdgeDict(ntypeDict, etypeKey, edgeTypeFile, edgeRelFile):
    edgeDict = {}
    for edgeType, edgeRel in zip(edgeTypeFile, edgeRelFile):
        edgeType = int(edgeType)
        edgeRel = tuple(map(int, edgeRel.strip().split(', ')))

        etypeRel = (ntypeDict[edgeRel[0]], 
                    etypeKey[edgeType], 
                    ntypeDict[edgeRel[1]])
        
        edgeRel = torch.tensor(edgeRel)

        if etypeRel not in edgeDict:
            edgeDict[etypeRel] = [edgeRel]
        else:
            edgeDict[etypeRel].append(edgeRel)

    return edgeDict


def createGraphDict(graphKey, edgeDict):
    graphDict = {}
    for etypeRel in edgeDict:
        for edgeRel in edgeDict[etypeRel]:
            graphID = graphKey[int(edgeRel[0])]
            if graphID not in graphDict:
                graphDict[graphID] = {etypeRel: [edgeRel]}
            else:
                if etypeRel not in graphDict[graphID]:
                    graphDict[graphID][etypeRel] = [edgeRel]
                else:
                    graphDict[graphID][etypeRel].append(edgeRel)

    return graphDict


def createLabeledGraphDict(graphDict, graphLabelFile):
    labeledGraphDict = {}
    for graphID, label in enumerate(graphLabelFile, 1):
        label = int(label)
        if label == -1:
            label = 0
        label = torch.tensor(label)
        labeledGraphDict[graphID] = [graphDict[graphID], label]
    
    return labeledGraphDict


def createNodeLabelsDict(graphLabels, graphIndicator, ntypeDict):
    labels = {}
    for i in range(len(ntypeDict)):
        ntype = ntypeDict[i + 1]
        graphID = graphIndicator[i] - 1
        nodeLabel = torch.tensor([graphLabels[graphID]])
        if ntype not in labels:
            labels[ntype] = nodeLabel
        else:
            labels[ntype] = torch.cat((labels[ntype], nodeLabel))
    
    return labels

processData()