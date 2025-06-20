import torch
import numpy as np
import networkx as nx
from src.dataset.instances.base import DataInstance
from src.dataset.instances.graph import GraphInstance
from src.core.explainer_base import Explainer
from src.utils.cfg_utils import retake_oracle
from src.utils.cfg_utils import retake_dataset
from operator import itemgetter




class GNN_MOExp(Explainer):
    
    def init(self):
        self.oracle = retake_oracle(self.local_config)
        self.dataset = retake_dataset(self.local_config)
        local_params = self.local_config['parameters']
        





    def explain(self, instance):

        # Prendere features dell'instance in questione
        node_features_matrix = instance.node_features

        # Creare grafo di networkx 
        G = nx.from_numpy_array(instance.data)
        
        # Estrarre prediction (y_i dell'articolo)
        softmax = torch.nn.Softmax(dim=0)
        log_softmax = torch.nn.LogSoftmax(dim=0)
        y_i = softmax(torch.tensor(self.oracle.predict_proba(instance)))
        log_y_i = log_softmax(torch.tensor(self.oracle.predict_proba(instance)))

        # nx.degree_centrality(grafo) = {indici_nodi : percentuali di nodi connessi}
        # Al nodo con degree massimo (che giocherà il ruolo del v_i dell'articolo) corrisponderà y_i
        degree_dict = nx.degree_centrality(G)

        # Selezionare nodo con degree massimo. selected_node_index lo individuerà nella lista G.nodes = [0, ..., num_nodes-1]
        selected_node_index = max(degree_dict, key=degree_dict.get)

        # Qui non c'è il num_conv_layers. Prendere L dal file di configurazione
        L = self.local_config['parameters']['max_depth']
        
        # DFS search crea sottografi ({G_i} dell'articolo) a partire da un nodo
        #G_i = nx.dfs_tree(G, source=selected_node_index, depth_limit=L) # Questo grafo è di tipo directed 
        #G_i = G_i.to_undirected() # Così è undirected

        # Per usare la rete neurale devi creare la GraphInstance. Estrarre le y dei grafi G e G_i_tilde
        # Porre la label delle nuove instances a zero per convenzione



        # Kullback-Leibler divergence
        KL_divergence = torch.nn.KLDivLoss(log_target=True, reduction='batchmean')





        # Definire i sottografi come nell'articolo (non sono tutti i sottografi possibili, ma solo quelli della DFS)
        dfs_edge_list = list(nx.dfs_edges(G, source=selected_node_index, depth_limit=L))
        
        # Sottografi esplorati durante il processo iterativo descritto nell'articolo salvati nella lista G_i_list
        # Il primo, cioè il grafo costituito dal solo nodo scelto, è aggiunto separatamente. C = limite numero nodi
        C = self.local_config['parameters']['max_num_nodes']
        first_subgraph = G.subgraph(selected_node_index)
        G_i_list = []
        G_i_list.append(first_subgraph)
        i = 0
        while i<len(dfs_edge_list):
            edges_in_subgraph = dfs_edge_list[:i+1]
            next_subgraph = nx.Graph()
            next_subgraph.add_edges_from(edges_in_subgraph)
            # Per porre il limite a C nodi come nell'articolo
            if len(next_subgraph.nodes)<=C:
                G_i_list.append(next_subgraph)
            i += 1
        
        # Costruzione della lista delle y_i_prime dell'articolo (nello spazio log) e delle liste delle metriche nu(G_i) e mu(G_i, G_i_tilde)
        nu_list = []
        mu_list = []
        log_y_i_prime_list = []
        i = 0
        while i<len(G_i_list):
            subgraph_adjacency_matrix = nx.adjacency_matrix(G_i_list[i]).toarray() 
            subgraph_node_features = [node_features_matrix[j] for j in G_i_list[i].nodes]
            subgraph_graph_instance = GraphInstance(id=-i, data=np.array(subgraph_adjacency_matrix), node_features=np.array(subgraph_node_features), label=0)
            
            #y_i_prime = softmax(self.oracle.predict_proba(subgraph_graph_instance)) 
            log_y_i_prime = log_softmax(torch.tensor(self.oracle.predict_proba(subgraph_graph_instance)))
            log_y_i_prime_list.append(log_y_i_prime)
            
            nu_G_i = -(KL_divergence(log_y_i, log_y_i_prime) + KL_divergence(log_y_i_prime, log_y_i))
            nu_list.append(nu_G_i)
            
            # Per ogni sottografo G_i studiare il set dei grafi precedenti (G_i_tilde, sottografi di G_i) nella DFS 
            # valutando la metrica mu(G_i, G_i_tilde). Ogni G_i è un possibile counterfactual per tutti i successivi nella DFS
            # Indice j scorre sui grafi precedenti nella DFS
            j = 0 
            # Saltare primo grafo (i>0)
            while i>0 and j<i:
                mu_G_i_G_i_tilde = (nu_list[i] - nu_list[j])/(len(G_i_list[i].edges) - len(G_i_list[j].edges)) 
                mu_list.append(mu_G_i_G_i_tilde)
                j += 1

            i += 1
        
        # Ricapitolando:
        # Si parte dall'instance G e da un suo nodo da spiegare
        # Si estrae un set di sottografi G_i tramite la DFS
        # Ad ogni G_i corrisponderanno ulteriori sottografi G_i_tilde, definiti come i precedenti nella DFS
        # Si formano abbinamenti (G_i, G_i_tilde) dove per ogni G_i ci sono più G_i_tilde
        # Valutare le metriche per ogni abbinamento e scegliere la migliore (Pareto optimal explanation)

        # Trovare Pareto optimal explanation. La lista candidates contiene tutte le possibili (G_i, G_i_tilde)
        # Ranking complessivo R=r1+r2 dove r1 ed r2 sono i ranking rispetto alle due metriche nu e mu
        # nu dipende solo da G_i, dunque tutte le (G_i, G_i_tilde) con G_i uguale hanno lo stesso ranking. Nella
        # lista nu_list sono stati inseriti in ordine
        candidates = []
        i = 1 # Saltare il primo perché non ha grafi precedenti
        while i<len(G_i_list):
            j = 0
            while j<i:
                candidate = [G_i_list[i], G_i_list[j], nu_list[i]] # Con prima metrica 
                mu_G_i_G_i_tilde = (nu_list[i] - nu_list[j])/(len(G_i_list[i].edges) - len(G_i_list[j].edges)) 
                candidate.append(mu_G_i_G_i_tilde) # Aggiungiamo la seconda metrica mu(G_i, G_i_tilde)       
                candidates.append(candidate)
                j += 1
            i += 1
        
        # candidates contiene liste di questa forma: [G_i, G_i_tilde, nu(G_i), mu(G_i, G_i_tilde)]
        # Ordinare rispetto alle metriche per ottenere ranking (ranking != indice) 

        # Per questa lista non è vero che ranking = indice perché G_i si ripetono 
        # La forma è
        #
        # [[G_i(1), G_i_tilde(0)=G_i(0), ...metriche...]
        #  [G_i(2), G_i_tilde(0)=G_i(0), ...metriche...]  
        #  [G_i(2), G_i_tilde(1)=G_i(1), ...metriche...]  
        #  ...
        #  [G_i(n), G_i_tilde(0)=G_i(0), ...metriche...]
        #  ...
        #  [G_i(n), G_i_tilde(n-1)=G_i(), ...metriche...]]
        #
        # Ad esempio, riguardo alla metrica 1 la seconda sublist ha lo stesso ranking r1 della terza perché 
        # la prima metrica (nu) dipende solo da G_i




        
        # Queste liste contengono i valori delle metriche. L'indice di nu(G_i) (mu(G_i, G_i_tilde)) è il ranking di 
        # (G_i, G_i_tilde) rispetto a nu (mu)
        ordered_nu_list = sorted(nu_list, reverse=True)
        ordered_mu_list = sorted(mu_list, reverse=True)

        # Aggiungere ad ogni sottolista di candidates il valore di R=r1+r2
        i = 0
        while i<len(candidates):
            candidates[i].append(ordered_nu_list.index(candidates[i][2]) + ordered_mu_list.index(candidates[i][3]))
            i += 1
         
        # Ordinare rispetto ad R (R basso = metriche elevate). R è sostanzialmente la posizione "in classifica"
        ordered_candidates = sorted(candidates, key=itemgetter(4))
        
        



        # Così tutte le righe con R=RMAX sono inserite
        explanations = [candidate_i for candidate_i in ordered_candidates if candidate_i[4]==ordered_candidates[0][4]]
        
        first_counterfactual_graph_instance = GraphInstance(id=0, data=np.array(nx.adjacency_matrix(explanations[0][1]).toarray()), node_features=np.array([node_features_matrix[j] for j in explanations[0][1].nodes]), label=0)
        print("Original graph predicted class = ", self.oracle.predict(instance))
        print("Counterfactual graph predicted class = ", self.oracle.predict(first_counterfactual_graph_instance))

        #i = 0
        #while i<len(explanations):
            #explanation = explanations[i] # Diverse righe possono avere lo stesso R=RMAX quindi inserirle tutte
            #print("Explanation graph G_i = ", explanation[0])
            #print("Counterfactual graph G_i_tilde = ", explanation[1])
            #print("Metric 1 nu(G_i) = ", float(explanation[2]))
            #print("Metric 2 mu(G_i, G_i_tilde) = ", float(explanation[3]))
            #print("Ranking R = ", explanation[4])
            #print("Original graph probability distribution y_i = ", y_i)
            #print("Counterfactual graph probability distribution = ", softmax(self.oracle.predict_proba(GraphInstance(id=0, data=np.array(nx.adjacency_matrix(explanation[1]).toarray()), node_features=np.array([node_features_matrix[j] for j in explanation[1].nodes]), label=0))))
            #i += 1        

        return first_counterfactual_graph_instance
    
    
    
    

    def check_configuration(self):
        super().check_configuration()
        local_config = self.local_config
        self.fold_id = self.local_config['parameters'].get('fold_id', -1)