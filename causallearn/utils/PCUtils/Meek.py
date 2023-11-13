from __future__ import annotations

from copy import deepcopy

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import sort_dict_ascending, test_obj, get_keys_from_value, get_keys_from_list_of_values

import itertools

def meek(cg_new: CausalGraph, background_knowledge: BackgroundKnowledge | None = None,
              verbose: bool = False) -> CausalGraph:
    """
    Run Meek rules

    Parameters
    ----------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    """

    # cg_new = deepcopy(cg)

    UT = cg_new.find_unshielded_triples()
    Tri = cg_new.find_triangles()
    Kite = cg_new.find_kites()

    loop = True

    while loop:
        loop = False
        if verbose:
            print("---------------- Processing Unshielded Triples ---------------- R1 Meek, 1995")
        for (i, j, k) in UT:
            if cg_new.is_fully_directed(i, j) and cg_new.is_undirected(j, k):
                if verbose:
                    print(f"{i} --> {j} and {j} -- {k}")
                if (background_knowledge is not None) and \
                        (background_knowledge.is_forbidden(cg_new.G.nodes[j], cg_new.G.nodes[k]) or
                         background_knowledge.is_required(cg_new.G.nodes[k], cg_new.G.nodes[j])):
                    pass
                else:
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[j], cg_new.G.nodes[k])
                    if edge1 is not None:
                        if cg_new.G.is_ancestor_of(cg_new.G.nodes[k], cg_new.G.nodes[j]):
                            if verbose:
                                # print(f"{cg_new.G.nodes[k].get_name()} is ancestor of {cg_new.G.nodes[j].get_name()}")
                                print(f"{k} is ancestor of {j}") ## NOTE: this check could be based on p-val
                            continue
                        else:
                            if verbose:
                                print(f"{k} is not ancestor of {j}")
                            cg_new.G.remove_edge(edge1)
                            ##TODO: add another rule here about anchestors
                    else:
                        continue
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[j], cg_new.G.nodes[k], Endpoint.TAIL, Endpoint.ARROW))
                    if verbose:
                        print(f'Oriented: j={j} --> k={k} ({cg_new.G.nodes[j].get_name()} --> {cg_new.G.nodes[k].get_name()})')
                    
                    ## Premise 1: UT 
                    premise_test1 = [test for test in cg_new.IKB_list if test.X=={i} and test.Y=={k} and \
                                        test.S==set() and test.dep_type=='I']
                    if premise_test1:
                        Premise1 = premise_test1[0] 
                    else:
                        Premise1 = test_obj(X={i}, S=set(), Y={k}, dep_type="I")
                        cg_new.IKB_list.append(Premise1)                      
                    ## Premise 2: (i, j, k) is NOT a V-structure 
                    premise_test2 = [test for test in cg_new.IKB_list if test.X=={i} and test.Y=={k} and \
                                        test.S=={j} and test.dep_type=='I']
                    if premise_test2:
                        Premise2 = premise_test2[0] 
                    else:
                        Premise2 = test_obj(X={i}, S={j}, Y={k}, dep_type="I")
                        cg_new.IKB_list.append(Premise2)  
                    Premise = (Premise1, Premise2, "arrow({}, {})".format(i, j))
                    Conclusion = "arrow({}, {})".format(j, k)
                    cg_new.decisions[(Premise)].add(Conclusion)

                    loop = True

        if verbose:
            print("---------------- Processing Triangles ---------------- R2 Meek, 1995")
        for (i, j, k) in Tri:
            if verbose:
                print((i, j, k),"is Tri")
            if cg_new.is_fully_directed(i, j) and cg_new.is_fully_directed(j, k) and cg_new.is_undirected(i, k):
                if verbose:
                    print(f"{i} --> {j} and {j} --> {k} and {i} -- {k}")
                if (background_knowledge is not None) and \
                        (background_knowledge.is_forbidden(cg_new.G.nodes[i], cg_new.G.nodes[k]) or
                         background_knowledge.is_required(cg_new.G.nodes[k], cg_new.G.nodes[i])):
                    pass
                else:
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[i], cg_new.G.nodes[k])
                    if edge1 is not None:
                        if cg_new.G.is_ancestor_of(cg_new.G.nodes[k], cg_new.G.nodes[i]):
                            continue
                        else:
                            cg_new.G.remove_edge(edge1)
                    else:
                        continue
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[i], cg_new.G.nodes[k], Endpoint.TAIL, Endpoint.ARROW))
                    if verbose:
                        print(f'Oriented: i={i} --> k={k} ({cg_new.G.nodes[i].get_name()} --> {cg_new.G.nodes[k].get_name()})')                                

                    ## Premise 2: Cannot have cycles and already have i --> j and j --> k                                           
                    Premise = tuple(["edge({}, {})".format(i, k), "arrow({}, {})".format(i, j), "arrow({}, {})".format(j, k)])
                    
                    ## Conclusion: Orient i-->k
                    Conclusion = "arrow({}, {})".format(i, k)
                    cg_new.decisions[(Premise)].add(Conclusion)

                    loop = True

        if verbose:
            print("---------------- Processing Kites ----------------")
        for (i, j, k, l) in Kite:
            if verbose:
                print((i, j, k, l),"is Kite")

            if cg_new.is_undirected(i, j) and cg_new.is_undirected(i, k) and cg_new.is_fully_directed(j, l) \
                    and cg_new.is_fully_directed(k, l) and cg_new.is_undirected(i, l):
                if verbose:
                    print(f"{i} -- {j} and {i} -- {k} and {j} --> {l} and {k} --> {l} and {i} -- {l}", "R3 Meek, 1995")
                if (background_knowledge is not None) and \
                        (background_knowledge.is_forbidden(cg_new.G.nodes[i], cg_new.G.nodes[l]) or
                         background_knowledge.is_required(cg_new.G.nodes[l], cg_new.G.nodes[i])):
                    pass
                else:
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[i], cg_new.G.nodes[l])
                    if edge1 is not None:
                        if cg_new.G.is_ancestor_of(cg_new.G.nodes[l], cg_new.G.nodes[i]):
                            continue
                        else:
                            cg_new.G.remove_edge(edge1)
                    else:
                        continue
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[i], cg_new.G.nodes[l], Endpoint.TAIL, Endpoint.ARROW))
                    if verbose:
                        print(f'Oriented: i={i} --> l={l} ({cg_new.G.nodes[i].get_name()} --> {cg_new.G.nodes[l].get_name()})')
                      
                    ## Additional Premise: Two edges are oriented
                    Premise = tuple(["edge({}, {})".format(i, k), "edge({}, {})".format(i, l), "arrow({}, {})".format(j, l), "arrow({}, {})".format(k, l)])
                    
                    ## Conclusion
                    Conclusion = "arrow({}, {})".format(i, l)
                    cg_new.decisions[(Premise)].add(Conclusion)
                    loop = True

            ### This rule only applies when the background knowledge is present
            # elif cg_new.is_undirected(i, j) and cg_new.is_undirected(i, k) and cg_new.is_fully_directed(l, j) \
            #         and cg_new.is_fully_directed(k, l) and (cg_new.is_fully_directed(i, l) or cg_new.is_fully_directed(l, i)):
            #     print(f"{i} -- {j} and {i} -- {k} and {l} --> {j} and {k} --> {l} and {i} o--o {l}", "R4 Meek, 1995")

            #     if (background_knowledge is not None) and \
            #             (background_knowledge.is_forbidden(cg_new.G.nodes[i], cg_new.G.nodes[l]) or
            #              background_knowledge.is_required(cg_new.G.nodes[l], cg_new.G.nodes[i])):
            #         pass
            #     else:
            #         edge1 = cg_new.G.get_edge(cg_new.G.nodes[i], cg_new.G.nodes[j])
            #         if edge1 is not None:
            #             if cg_new.G.is_ancestor_of(cg_new.G.nodes[j], cg_new.G.nodes[i]):
            #                 continue
            #             else:
            #                 cg_new.G.remove_edge(edge1)
            #         else:
            #             continue
            #         cg_new.G.add_edge(Edge(cg_new.G.nodes[i], cg_new.G.nodes[j], Endpoint.TAIL, Endpoint.ARROW))
            #         print(f'Oriented: i={i} --> l={j} ({cg_new.G.nodes[i].get_name()} --> {cg_new.G.nodes[j].get_name()})')

            #         ## Additional Premise: Three edges are oriented
            #         if cg_new.is_fully_directed(i, l):
            #             third_dir_edge = ["arrow({}, {})".format(i, l)]
            #         elif cg_new.is_fully_directed(l, i):
            #             third_dir_edge = ["arrow({}, {})".format(l, i)]
            #         Premise = kite_prem + ["arrow({}, {})".format(l, j), "arrow({}, {})".format(k, l)] + third_dir_edge

            #         ## Conclusion
            #         Conclusion = "arrow({}, {})".format(i, j)
            #         cg_new.decisions[(Premise)].add(Conclusion)
            #         loop = True
    return cg_new


def definite_meek(cg: CausalGraph, background_knowledge: BackgroundKnowledge | None = None) -> CausalGraph:
    """
    Run Meek rules over the definite unshielded triples

    Parameters
    ----------
    cg : a CausalGraph object. Where cg.G.graph[j,i]=1 and cg.G.graph[i,j]=-1 indicates  i --> j ,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = -1 indicates i --- j,
                    cg.G.graph[i,j] = cg.G.graph[j,i] = 1 indicates i <-> j.
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    """

    cg_new = deepcopy(cg)

    Tri = cg_new.find_triangles()
    Kite = cg_new.find_kites()

    loop = True

    while loop:
        loop = False
        for (i, j, k) in cg_new.definite_non_UC:
            if cg_new.is_fully_directed(i, j) and \
                    cg_new.is_undirected(j, k) and \
                    not ((background_knowledge is not None) and
                         (background_knowledge.is_forbidden(cg_new.G.nodes[j], cg_new.G.nodes[k]) or
                          background_knowledge.is_required(cg_new.G.nodes[k], cg_new.G.nodes[j]))):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[j], cg_new.G.nodes[k])
                if edge1 is not None:
                    if cg_new.G.is_ancestor_of(cg_new.G.nodes[k], cg_new.G.nodes[j]):
                        continue
                    else:
                        cg_new.G.remove_edge(edge1)
                else:
                    continue
                cg_new.G.add_edge(Edge(cg_new.G.nodes[j], cg_new.G.nodes[k], Endpoint.TAIL, Endpoint.ARROW))
                loop = True
            elif cg_new.is_fully_directed(k, j) and \
                    cg_new.is_undirected(j, i) and \
                    not ((background_knowledge is not None) and
                         (background_knowledge.is_forbidden(cg_new.G.nodes[j], cg_new.G.nodes[i]) or
                          background_knowledge.is_required(cg_new.G.nodes[i], cg_new.G.nodes[j]))):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[j], cg_new.G.nodes[i])
                if edge1 is not None:
                    if cg_new.G.is_ancestor_of(cg_new.G.nodes[i], cg_new.G.nodes[j]):
                        continue
                    else:
                        cg_new.G.remove_edge(edge1)
                else:
                    continue
                cg_new.G.add_edge(Edge(cg_new.G.nodes[j], cg_new.G.nodes[i], Endpoint.TAIL, Endpoint.ARROW))
                loop = True

        for (i, j, k) in Tri:
            if cg_new.is_fully_directed(i, j) and cg_new.is_fully_directed(j, k) and cg_new.is_undirected(i, k):
                if (background_knowledge is not None) and \
                        (background_knowledge.is_forbidden(cg_new.G.nodes[i], cg_new.G.nodes[k]) or
                         background_knowledge.is_required(cg_new.G.nodes[k], cg_new.G.nodes[i])):
                    pass
                else:
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[i], cg_new.G.nodes[k])
                    if edge1 is not None:
                        if cg_new.G.is_ancestor_of(cg_new.G.nodes[k], cg_new.G.nodes[i]):
                            continue
                        else:
                            cg_new.G.remove_edge(edge1)
                    else:
                        continue
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[i], cg_new.G.nodes[k], Endpoint.TAIL, Endpoint.ARROW))
                    loop = True

        for (i, j, k, l) in Kite:
            if ((j, l, k) in cg_new.definite_UC or (k, l, j) in cg_new.definite_UC) \
                    and ((j, i, k) in cg_new.definite_non_UC or (k, i, j) in cg_new.definite_non_UC) \
                    and cg_new.is_undirected(i, l):
                if (background_knowledge is not None) and \
                        (background_knowledge.is_forbidden(cg_new.G.nodes[i], cg_new.G.nodes[l]) or
                         background_knowledge.is_required(cg_new.G.nodes[l], cg_new.G.nodes[i])):
                    pass
                else:
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[i], cg_new.G.nodes[l])
                    if edge1 is not None:
                        if cg_new.G.is_ancestor_of(cg_new.G.nodes[l], cg_new.G.nodes[i]):
                            continue
                        else:
                            cg_new.G.remove_edge(edge1)
                    else:
                        continue
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[i], cg_new.G.nodes[l], Endpoint.TAIL, Endpoint.ARROW))
                    loop = True

    return cg_new
