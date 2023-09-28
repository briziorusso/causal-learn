from __future__ import annotations

from itertools import combinations

import numpy as np
from numpy import ndarray
from typing import List
from tqdm.auto import tqdm

from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import append_value, test_obj
from causallearn.utils.cit import CIT


def skeleton_discovery(
    data: ndarray, 
    alpha: float, 
    indep_test: CIT,
    stable: bool = True,
    keep_edges: bool = False,
    ikb: bool = False,
    background_knowledge: BackgroundKnowledge | None = None, 
    verbose: bool = False,
    show_progress: bool = True,
    node_names: List[str] | None = None, 
) -> CausalGraph:
    """
    Perform skeleton discovery

    Parameters
    ----------
    data : data set (numpy ndarray), shape (n_samples, n_features). The input data, where n_samples is the number of
            samples and n_features is the number of features.
    alpha: float, desired significance level of independence tests (p_value) in (0,1)
    indep_test : class CIT, the independence test being used
            [fisherz, chisq, gsq, mv_fisherz, kci]
           - fisherz: Fisher's Z conditional independence test
           - chisq: Chi-squared conditional independence test
           - gsq: G-squared conditional independence test
           - mv_fisherz: Missing-value Fishers'Z conditional independence test
           - kci: Kernel-based conditional independence test
    stable : run stabilized skeleton discovery if True (default = True)
    background_knowledge : background knowledge
    verbose : True iff verbose output should be printed.
    show_progress : True iff the algorithm progress should be show in console.
    node_names: Shape [n_features]. The name for each feature (each feature is represented as a Node in the graph, so it's also the node name)

    Returns
    -------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=0 and cg.G.graph[i,j]=1 indicates  i -> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i -- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.

    """

    assert type(data) == np.ndarray
    assert 0 < alpha < 1

    no_of_var = data.shape[1]
    cg = CausalGraph(no_of_var, node_names)
    cg.set_ind_test(indep_test)
    cg.alpha = alpha

    depth = -1
    pbar = tqdm(total=no_of_var) if show_progress else None
    while cg.max_degree() - 1 > depth:
        depth += 1
        edge_removal = []
        if show_progress:
            pbar.reset()
        for x in range(no_of_var):
            if show_progress:
                pbar.update()
            if show_progress:
                pbar.set_description(f'Depth={depth}, working on node {x}')
            if verbose:
                print(f'Depth={depth}, working on node {cg.G.nodes[x].get_name()}')
            Neigh_x = cg.neighbors(x)
            if len(Neigh_x) < depth - 1:
                continue
            for y in Neigh_x:
                knowledge_ban_edge = False
                sepsets = set()
                if background_knowledge is not None and (
                        background_knowledge.is_forbidden(cg.G.nodes[x], cg.G.nodes[y])
                        and background_knowledge.is_forbidden(cg.G.nodes[y], cg.G.nodes[x])):
                    knowledge_ban_edge = True
                if knowledge_ban_edge:
                    if not stable:
                        edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                        if edge1 is not None:
                            cg.G.remove_edge(edge1)
                        edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                        if edge2 is not None:
                            cg.G.remove_edge(edge2)
                        append_value(cg.sepset, x, y, ())
                        append_value(cg.sepset, y, x, ())
                        break
                    else:
                        if not keep_edges:
                            edge_removal.append((x, y))  # after all conditioning sets at
                            edge_removal.append((y, x))  # depth l have been considered

                Neigh_x_noy = np.delete(Neigh_x, np.where(Neigh_x == y))
                for S in combinations(Neigh_x_noy, depth):
                    p = cg.ci_test(x, y, S)
                    if p > alpha:
                        if verbose:
                            print('%s _||_ %s | %s with p-value %f\n' % (cg.G.nodes[x].get_name(), cg.G.nodes[y].get_name(), [cg.G.nodes[s].get_name() for s in S], p))
                        if not stable:
                            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
                            if edge1 is not None:
                                cg.G.remove_edge(edge1)
                            edge2 = cg.G.get_edge(cg.G.nodes[y], cg.G.nodes[x])
                            if edge2 is not None:
                                cg.G.remove_edge(edge2)
                            append_value(cg.sepset, x, y, S)
                            append_value(cg.sepset, y, x, S)
                            break
                        else:
                            if not keep_edges:
                                edge_removal.append((x, y))  # after all conditioning sets at
                                edge_removal.append((y, x))  # depth l have been considered
                            for s in S:
                                sepsets.add(s)
                        append_value(cg.IKB, x, y, (S,p))
                        append_value(cg.IKB, y, x, (S,p))
                        if ikb:
                            cg.IKB_list.append(test_obj(X={x},S=set(S),Y={y},p_val=p,alpha=alpha))

                    else:
                        if verbose:
                            print('%s _|/|_ %s | %s with p-value %f\n' % (cg.G.nodes[x].get_name(), cg.G.nodes[y].get_name(), [cg.G.nodes[s].get_name() for s in S], p))
                        append_value(cg.IKB, x, y, (S,p))
                        append_value(cg.IKB, y, x, (S,p))
                        if ikb:
                            cg.IKB_list.append(test_obj(X={x},S=set(S),Y={y},p_val=p,alpha=alpha))
                append_value(cg.sepset, x, y, tuple(sepsets))
                append_value(cg.sepset, y, x, tuple(sepsets))

        if show_progress:
            pbar.refresh()

        for (x, y) in list(set(edge_removal)):
            if verbose:
                print('Removing %s -- %s. Sepset: %s p-value %s\n' % (cg.G.nodes[x].get_name(), cg.G.nodes[y].get_name(), cg.sepset[x,y],  cg.IKB[x,y]))                
            edge1 = cg.G.get_edge(cg.G.nodes[x], cg.G.nodes[y])
            if edge1 is not None:
                cg.G.remove_edge(edge1)
                
                if not keep_edges: ##NOTE: check this for argumentative causal discovery
                    try:
                        Premise = [test for test in cg.IKB_list if test.X=={x} and test.Y=={y} and test.S==set(S)][0]
                        Conclusion = "notedge({}, {})".format(x, y)
                        cg.decisions[tuple([Premise])].add(Conclusion)
                    except:
                        pass

    # cg.draw_pydot_graph()

    if show_progress:
        pbar.close()

    return cg
