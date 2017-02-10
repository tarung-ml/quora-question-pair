# Perform clustering on the words (using cosine-distance) and visualize

from gensim.models import Word2Vec
from nltk.corpus import stopwords

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import numpy as np, pandas as pd
from sklearn.cluster import AgglomerativeClustering

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
plt.interactive(False) # maybe pycharm specific

from collections import Counter

def word_count(model, word):
    return model.vocab[word].count if word in model.vocab else 0

# just visualizing the frequent words (not scalable)
def writeGraphBasic(model):
    wordCounts = [(word, word_count(model, word)) for word in model.vocab]
    # remove stopwords and low-freq and char
    s = set(stopwords.words('english'))
    freqWords = [word[0] for word in wordCounts if word[1] > minCount and word[0] not in s and len(word[0]) > 1]
    G = nx.Graph()
    for i in range(len(freqWords)):
        for j in range(i + 1, len(freqWords)):
            #print(freqWords[i], freqWords[j])
            G.add_edge(freqWords[i], freqWords[j], weight=model.similarity(freqWords[i], freqWords[j]))

    nx.write_gpickle(G, "model/G.gpickle")


# try sum of within cluster-variance as a measure of goodness
def withinSS(feats_vec, labels, centers):
    result = 0
    for i in range(feats_vec.shape[0]):
        result += np.sum((feats_vec[i,] - centers[labels[i]])**2)
    return result


# if want to try dimensionality reduction
def spectralClustering(X):
    # build similarity matrix
    """diagonal = X.apply(lambda x: 1/np.sqrt(sum(x**2)), axis = 1)
    diagonal = np.diag(diagonal)
    diagonal[(~np.isfinite(diagonal))] = 1
    sim_mat_raw = diagonal.dot(X.dot(X.transpose())).dot(diagonal)
    # decompose the feature-feature similarity matrix via pca and retain top K components (contains ~ 80-85% of energy)
    # find optimal K
    pca = PCA()
    pca.fit(sim_mat_raw)
    explainedVar = np.cumsum(pca.explained_variance_ratio_)
    n_comps = next(x for x in range(len(explainedVar)) if explainedVar[x] > 0.80)
    print("n_comps" + str(n_comps))
    #Looking at above plot
    pca = PCA(n_components=n_comps)
    pca.fit(sim_mat_raw)
    feats_vec = pca.fit_transform(sim_mat_raw)"""
    feats_vec = X
    # cluster the feature-vectors and obtain "soft" cluster memberships
    # find optimal C
    lowest_withinSS = np.infty
    withinSSList = []
    n_components_range = range(1000, feats_vec.shape[0], 1000)
    # n_components_range = range(2, 10)
    for n_components in n_components_range:
        print(n_components, withinSSList)
        # Fit a mixture of Gaussians with EM
        kmeans = KMeans(n_clusters=n_components, random_state=1)
        kmeans.fit(feats_vec)
        labels = kmeans.labels_
        withinSSList.append(withinSS(feats_vec, labels, kmeans.cluster_centers_))
        if withinSSList[-1] < lowest_withinSS:
            lowest_withinSS = withinSSList[-1]
            best_kmm = kmeans
    return best_kmm



# try clustering the words and sample some from each cluster to capture strong cluster-memberships
def writeGraph(model, df):
    #best = spectralClustering(df.ix[:, 1:].as_matrix()); print(best)
    kmeans = AgglomerativeClustering(n_clusters = N_CLUSTERS, affinity='cosine', linkage="complete") #spectralClustering(df.ix[:,1:]) #
    kmeans.fit_predict(df.ix[:,1:])
    labels = kmeans.labels_
    # identify words within each cluster
    clustered_words = []
    print(len(labels))
    words = list(df.ix[:,0])
    #top_labels = [x[0] for x in sorted(Counter(labels).items(), key = lambda x: x[1], reverse = True)]
    for cluster in set(labels):
        candidates = [index for index in np.where(labels == cluster)[0]]
        candidate_counts =[(words[index], word_count(model, words[index])) for index in candidates]
        candidate_counts = sorted(candidate_counts, key = lambda x: x[1], reverse = True)[0: int(SAMPLE*len(candidates))]
        clustered_words += candidate_counts

    # remove stopwords and low-freq and chars
    s = set(stopwords.words('english'))
    clustered_words = [word[0] for word in clustered_words if word[1] > minCount and word[0] not in s and len(word[0]) > 1]
    print(len(clustered_words))
    G = nx.Graph()
    for i in range(len(clustered_words)):
        for j in range(i + 1, len(clustered_words)):
            if i % 100 == 0 and j == 0: print(i)
            G.add_edge(clustered_words[i], clustered_words[j], weight=model.similarity(clustered_words[i], clustered_words[j]))

    nx.write_gpickle(G, "model/G.gpickle")

def removeStrayNodes(G):
    # find strongly connected components
    H = nx.connected_component_subgraphs(G)
    subgraphs = [ (g, len(g.nodes())) for g in H] #sum([word_count(model, node) for node in g.nodes()])
    subgraphs = sorted(subgraphs, key = lambda x: x[1], reverse = True)
    subgraphs = [subgraph for subgraph in subgraphs if subgraph[1] > MIN_SIZE_SUBGRAPH]
    connected_nodes = [node for subgraph in subgraphs
                       if sum([word_count(model, node)
                               for node in subgraph[0].nodes()])> len(subgraph[0].nodes())*minCount*2 for node in subgraph[0].nodes()]
    G.remove_nodes_from([node for node in G.nodes() if node not in connected_nodes])

def graphToPdf(model):
    G = nx.read_gpickle("model/G.gpickle")
    G.remove_edges_from([(a, b) for (a, b, w) in G.edges(data=True) if w['weight'] < minSim])
    removeStrayNodes(G) #G.remove_nodes_from([node for node in G.nodes() if G.degree(node) < 2])
    print("viz node count: " + str(len(G.nodes())))
    edges = G.edges(data=True)
    weights = [1/(1 - w['weight'])/10 for (a, b, w) in edges]
    # draw full graph
    nx.draw(G, graphviz_layout(G, prog="neato"), with_labels=True, font_size=4,
            node_size=[int(0.04* word_count(model, node)) for node in G.nodes()], alpha=0.3, linewidths=0.01,
            width=weights, node_color='b', edge_color='r', edges = edges)
    plt.draw(); plt.savefig("plots/full.pdf", bbox_inches="tight"); plt.clf(); plt.close()
    # draw strongly connected components
    H = nx.connected_component_subgraphs(G)
    subgraphs = [ (g, len(g.nodes())) for g in H] #sum([word_count(model, node) for node in g.nodes()])
    subgraphs = sorted(subgraphs, key = lambda x: x[1], reverse = True)
    print(subgraphs)
    count  = 0
    for subgraph in subgraphs[0:SHOW_SUBGRAPHS]:
        G = subgraph[0]
        edges = G.edges(data = True)
        weights = [1/(1 - w['weight'])/10 for (a, b, w) in edges];
        nx.draw(G, graphviz_layout(G, prog="neato"), with_labels=True, font_size=5,
                node_size=[int(0.2 * word_count(model, node)) for node in G.nodes()], alpha=0.3, linewidths=0.01,
                width=weights, node_color='b', edge_color='r', edges = edges)
        plt.draw(); plt.savefig("plots/subgraphs_" + str(count) +".pdf", bbox_inches="tight");plt.clf(); plt.close()
        count += 1



minCount = 200
minSim = 0.8
SHOW_SUBGRAPHS = 10
MIN_SIZE_SUBGRAPH = 3
# if sampling, set below
N_CLUSTERS = 20
SAMPLE = 0.2


if __name__== '__main__':
    model = Word2Vec.load("model/word2vec_model_bigrams")
    writeGraphBasic(model)
    df = pd.read_csv("model/word2vec_model_bigrams.csv")
    #writeGraph(model, df)
    graphToPdf(model)