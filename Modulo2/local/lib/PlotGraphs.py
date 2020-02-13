import numpy as np 
import string 
import networkx as nx
import matplotlib.pyplot as plt



def PlotUndirectedGraph(A):
    NodesNames = list(string.ascii_uppercase);
    NNodes = A.shape[0]
    G = nx.DiGraph()
    for i in range(NNodes):
        G.add_node(NodesNames[i])
    for i in range(NNodes):
        for j in range(i+1,NNodes):
            if A[i,j] != 0:
                G.add_edge(NodesNames[i],NodesNames[j],weight=A[i,j])
    pos = nx.spring_layout(G)
    edge_labels=dict([((u,v,),d['weight'])
                 for u,v,d in G.edges(data=True)])
    #edge_colors = ['black' if not edge in red_edges else 'red' for edge in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color = 'skyblue')
    #nx.draw(G,pos, node_color = values, node_size=1500,edge_color=edge_colors,edge_cmap=plt.cm.Reds)

    nx.draw_networkx_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx_edges(G, pos, arrows = False)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()