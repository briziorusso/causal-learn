from __future__ import annotations

from copy import deepcopy

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import append_value, sort_dict_ascending, test_obj, get_keys_from_value, get_keys_from_list_of_values
import math, statistics
import numpy as np
import pandas as pd

import igraph as ig
def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()

def uc_sepset(cg_new: CausalGraph, priority: int = 3,
              background_knowledge: BackgroundKnowledge | None = None,
              verbose: bool = False) -> CausalGraph:
    """
    Run (UC_sepset) to orient unshielded colliders

    Parameters
    ----------
    cg : a CausalGraph object
    priority : rule of resolving conflicts between unshielded colliders (default = 3)
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    """

    assert priority in [0, 1, 2, 3, 4]

    # cg_new = deepcopy(cg)

    R0 = []  # Records of possible orientations
    UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        if verbose:
            print((x,y,z))
            print(f"(X{x+1},X{y+1},X{z+1})")
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                 background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue
        if all(y not in S for S in cg_new.sepset[x, z]): ##NOTE: this is CPC, double check in pcalg
            if verbose:
                print(f"all(y={y} not in S for S in cg.sepset[{x}, {z}]={cg_new.sepset[x, z]})")
            
            if priority == 0:  # 0: overwrite
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                edge2 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[x])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                # Fully orient the edge irrespective of what have been oriented
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                edge3 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[z])
                if edge3 is not None:
                    cg_new.G.remove_edge(edge3)
                edge4 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge4 is not None:
                    cg_new.G.remove_edge(edge4)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 1:  # 1: orient bi-directed
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    if cg_new.G.graph[x, y] == Endpoint.TAIL.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[x, y] == Endpoint.ARROW.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    if cg_new.G.graph[z, y] == Endpoint.TAIL.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[z, y] == Endpoint.ARROW.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 2:  # 2: prioritize existing
                if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    # Orient only if the edges have not been oriented the other way around
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    if verbose:
                        print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                    edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                    if edge2 is not None:
                        cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    if verbose:
                        print(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')

            else:
                R0.append((x, y, z))

    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
            for (x, y, z) in R0:
                cond = cg_new.find_cond_sets_with_mid(x, z, y)
                UC_dict[(x, y, z)] = max([cg_new.ci_test(x, z, S) for S in cond])
            UC_dict = sort_dict_ascending(UC_dict)

        else:  # 4. Order colliders by p_{xy|not y} in descending order
            for (x, y, z) in R0:
                cond = cg_new.find_cond_sets_without_mid(x, z, y)
                UC_dict[(x, y, z)] = max([cg_new.ci_test(x, z, S) for S in cond])
            UC_dict = sort_dict_ascending(UC_dict, descending=True)

        for (x, y, z) in UC_dict.keys():
            if verbose:
                print((x,y,z))
                print(f"(X{x+1},X{y+1},X{z+1})")
            if (background_knowledge is not None) and \
                    (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                     background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
                continue
            if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                # Orient only if the edges have not been oriented the other way around
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    print(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')

        return cg_new


def maxp(cg_new: CausalGraph, priority: int = 3, background_knowledge: BackgroundKnowledge = None,
              ikb: bool = False, verbose: bool = False):
    """
    Run (MaxP) to orient unshielded colliders

    Parameters
    ----------
    cg : a CausalGraph object
    priority : rule of resolving conflicts between unshielded colliders (default = 3)
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    """

    assert priority in [0, 1, 2, 3, 4]
    maxp_rule = False

    # cg_new = deepcopy(cg)
    UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        if verbose:
            print((x, y, z))
            print(f"(X{x+1},X{y+1},X{z+1})")
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                 background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue

        cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
        # cond_with_y_p = {S:cg_new.ci_test(x, z, S) for S in cond_with_y}
        if verbose:
            print(f"cond_with_y: {cond_with_y_p}")
        # ikb_dict_x_z = {k: v for d in cg_new.IKB[x,z] for k, v in d.items()}
        ## Add additional test to IKB_list if not already there
        if ikb:
            cg_new.IKB_list = cg_new.IKB_list + \
                [test_obj(X={x},S=set(S),Y={z},p_val=cg_new.ci_test(x, z, S),alpha=cg_new.alpha) for S in cond_with_y if S not in \
                    [tuple(test.S) for test in cg_new.IKB_list if test.X=={x} and test.Y=={z}]]
        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)
        # cond_without_y_p = {S:cg_new.ci_test(x, z, S) for S in cond_without_y}
        if verbose:
            print(f"cond_without_y:{cond_without_y_p}")
        ## Add additional test to IKB_list if not already there
        if ikb:
            cg_new.IKB_list = cg_new.IKB_list + \
                [test_obj(X={x},S=set(S),Y={z},p_val=cg_new.ci_test(x, z, S),alpha=cg_new.alpha) for S in cond_without_y if S not in 
                    [tuple(test.S) for test in cg_new.IKB_list if test.X=={x} and test.Y=={z}]]

        max_p_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_with_y])
        max_p_not_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_without_y])

        if max_p_not_contain_y > max_p_contain_y:
            if verbose:
                print("max_p_not_contain_y > max_p_contain_y",max_p_not_contain_y, max_p_contain_y)

            if priority == 0:  # 0: overwrite
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                edge2 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[x])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                # Fully orient the edge irrespective of what have been oriented
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')

                edge3 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[z])
                if edge3 is not None:
                    cg_new.G.remove_edge(edge3)
                edge4 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge4 is not None:
                    cg_new.G.remove_edge(edge4)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    print(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')

            elif priority == 1:  # 1: orient bi-directed
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    if cg_new.G.graph[x, y] == Endpoint.TAIL.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[x, y] == Endpoint.ARROW.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    if cg_new.G.graph[z, y] == Endpoint.TAIL.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[z, y] == Endpoint.ARROW.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 2:  # 2: prioritize existing
                if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    # Orient only if the edges have not been oriented the other way around
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    if verbose:
                        print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')

                    edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                    if edge2 is not None:
                        cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    if verbose:
                        print(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')

            elif priority == 3:
                UC_dict[(x, y, z)] = max_p_contain_y
                if verbose:
                    print(f"Chosen: {[{S,cg_new.ci_test(x, z, S)} for S in cond_with_y if cg_new.ci_test(x, z, S)==max_p_contain_y]}")

            elif priority == 4:
                UC_dict[(x, y, z)] = max_p_not_contain_y
                if verbose:
                    print(f"Chosen: {[{S,cg_new.ci_test(x, z, S)} for S in cond_without_y if cg_new.ci_test(x, z, S)==max_p_not_contain_y]}")

                ## Rule based on MaxP. remove for now
                # if maxp_rule == True:
                #     loosing_cond_set = get_keys_from_value(cond_with_y_p, max_p_contain_y)
                #     winning_cond_set = get_keys_from_value(cond_without_y_p, max_p_not_contain_y)
                #     # collider = set(loosing_cond_set) - set(winning_cond_set) #TODO: revise for lenght issues

                #     ## Premises 
                #     # x _||_ z #
                #     ### This is the condition for being UT in the first place
                #     UC_premise_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                #                         test.S==set() and test.dep_type=='I'][0]                
                #     # x _||_ z | {W} #
                #     ### The strongest test does not contain y, hence independence is attributed to {W} 
                #     # and y considered a collider
                #     premise_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                #                         test.S==set(winning_cond_set) and test.dep_type=='I']
                #     if premise_test:
                #         Premise = premise_test[0] 
                #     else:
                #         Premise = test_obj(X={x}, S=set(winning_cond_set), Y={z}, dep_type="I")
                #         cg_new.IKB_list.append(Premise)
                    
                #     ## Conclusions
                #     # x _|/|_ z | y ### y is collider
                #     conc1_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                #                         test.S=={y} and test.dep_type=='D']
                #     if conc1_test:
                #         Conclusion1 = conc1_test[0] 
                #     else:
                #         Conclusion1 = test_obj(X={x}, S={y}, Y={z}, dep_type="D")
                #         cg_new.IKB_list.append(Conclusion1)

                #     # x _|/|_ z | {W} + y ### Weaker test is not "believed" hence made a dependence 
                #     # even though returns 'I' from the data
                #     conc2_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                #                         test.S==set(loosing_cond_set) and test.dep_type=='D']
                #     if conc2_test:
                #         Conclusion2 = conc2_test[0] 
                #     else:
                #         Conclusion2 = test_obj(X={x}, S=set(loosing_cond_set), Y={z}, dep_type="D")
                #         cg_new.IKB_list.append(Conclusion2)
                    
                #     full_premise = (UC_premise_test, Premise) if UC_premise_test != Premise else tuple([Premise])
                #     full_conclusion = (Conclusion1, Conclusion2) if Conclusion1 != Conclusion2 else tuple([Conclusion1])

                #     cg_new.decisions[full_premise].add(full_conclusion)
        # else:
            # if maxp_rule:
            #     ##TODO: do I need this?
            #     if verbose:
            #         print("max_p_not_contain_y <= max_p_contain_y",max_p_not_contain_y, max_p_contain_y)
            #     loosing_cond_set = get_keys_from_value(cond_without_y_p, max_p_not_contain_y)
            #     winning_cond_set = get_keys_from_value(cond_with_y_p, max_p_contain_y)
            #     # non_collider = set(loosing_cond_set) - set(winning_cond_set) #TODO: revise for lenght issues

            #     ## Premises 
            #     # x _||_ z #
            #     ### This is the condition for being UT in the first place
            #     UC_premise_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
            #                         test.S==set() and test.dep_type=='I'][0]                
            #     # x _||_ z | {W} + y #
            #     ### The strongest test contains y, hence independence is attributed to y
            #     # and y is NOT considered a collider
            #     premise_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
            #                         test.S==set(winning_cond_set) and test.dep_type=='I']
            #     if premise_test:
            #         Premise = premise_test[0] 
            #     else:
            #         Premise = test_obj(X={x}, S=set(winning_cond_set), Y={z}, dep_type="I")
            #         cg_new.IKB_list.append(Premise)

            #     ## Conclusions
            #     # x _||_ z | y ### y is NOT a collider
            #     conc1_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
            #                         test.S=={y} and test.dep_type=='I']
            #     if conc1_test:
            #         Conclusion1 = conc1_test[0] 
            #     else:
            #         Conclusion1 = test_obj(X={x}, S={y}, Y={z}, dep_type="I")
            #         cg_new.IKB_list.append(Conclusion1)

            #     conc2_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
            #                         test.S==set(loosing_cond_set) and test.dep_type=='D']

            #     # x _|/|_ z | {W} ### Weaker test is not "believed" hence made a dependence 
            #     # even though returns I solely from the data
            #     if conc2_test:
            #         Conclusion2 = conc2_test[0] 
            #     else:
            #         Conclusion2 = test_obj(X={x}, S=set(loosing_cond_set), Y={z}, dep_type="D")
            #         cg_new.IKB_list.append(Conclusion2)

            #     full_premise = (UC_premise_test, Premise) if UC_premise_test != Premise else tuple([Premise])
            #     full_conclusion = (Conclusion1, Conclusion2) if Conclusion1 != Conclusion2 else tuple([Conclusion1])

            #     cg_new.decisions[full_premise].add(full_conclusion)

    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
            UC_dict = sort_dict_ascending(UC_dict)
            if verbose:
                print('UC_dict', UC_dict)
        else:  # 4. Order colliders by p_{xz|not y} in descending order
            UC_dict = sort_dict_ascending(UC_dict, True)
            if verbose:
                print('UC_dict', UC_dict)

        for (x, y, z) in UC_dict.keys():
            if verbose:
                print((x,y,z))
            if (background_knowledge is not None) and \
                    (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                     background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
                continue

            if maxp_rule:
                ### Premises
                ## UT condition
                UC_premise_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                                    test.S==set() and test.dep_type=='I'][0]
                ## Collider Condition
                premise_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                                    test.S=={y} and test.dep_type=='D']
                if premise_test:
                    Premise = premise_test[0] 
                else:
                    ### Add to IKB_list if not already there - this is an inferred test from the maxp rule
                    Premise = test_obj(X={x}, S={y}, Y={z}, dep_type="D")
                    cg_new.IKB_list.append(Premise)  

            if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                # Orient only if the edges have not been oriented the other way around
                ##NOTE shall I code this condition too?
                if verbose:
                    print(f"{y} -- {x} and {y} -- {z}")

                ### Conclusions: Orient V-structure               
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')   
                Conclusion1 = "arrow({}, {})".format(x, y)

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    print(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                Conclusion2 = "arrow({}, {})".format(z, y)
                if maxp_rule:
                    cg_new.decisions[(UC_premise_test, Premise)].add((Conclusion1, Conclusion2))

            elif maxp_rule:
                #remove from decisions if edge is not removed
                del cg_new.decisions[get_keys_from_value(cg_new.decisions, (UC_premise_test, Premise))]

        return cg_new


def definite_maxp(cg: CausalGraph, alpha: float, priority: int = 4,
                  background_knowledge: BackgroundKnowledge = None) -> CausalGraph:
    """
    Run (Definite_MaxP) to orient unshielded colliders

    Parameters
    ----------
    cg : a CausalGraph object
    priority : rule of resolving conflicts between unshielded colliders (default = 3)
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    """

    assert 1 > alpha >= 0
    assert priority in [2, 3, 4]

    cg_new = deepcopy(cg)
    UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        print((x,y,z))
        cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
        print(f"cond_with_y: {cond_with_y}")
        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)
        print(f"cond_without_y: {cond_without_y}")
        max_p_contain_y = 0
        max_p_not_contain_y = 0
        uc_bool = True
        nuc_bool = True

        for S in cond_with_y:
            p = cg_new.ci_test(x, z, S)
            if p > alpha:
                uc_bool = False
                break
            elif p > max_p_contain_y:
                max_p_contain_y = p

        for S in cond_without_y:
            p = cg_new.ci_test(x, z, S)
            if p > alpha:
                nuc_bool = False
                if not uc_bool:
                    break  # ambiguous triple
            if p > max_p_not_contain_y:
                max_p_not_contain_y = p

        if uc_bool:
            if nuc_bool:
                if max_p_not_contain_y > max_p_contain_y:
                    print("max_p_not_contain_y > max_p_contain_y",max_p_not_contain_y, max_p_contain_y)
                    if priority in [2, 3]:
                        UC_dict[(x, y, z)] = max_p_contain_y
                    if priority == 4:
                        UC_dict[(x, y, z)] = max_p_not_contain_y
                else:
                    cg_new.definite_non_UC.append((x, y, z))
            else:
                if priority in [2, 3]:
                    UC_dict[(x, y, z)] = max_p_contain_y
                if priority == 4:
                    UC_dict[(x, y, z)] = max_p_not_contain_y

        elif nuc_bool:
            cg_new.definite_non_UC.append((x, y, z))

    if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
        UC_dict = sort_dict_ascending(UC_dict)
        print(UC_dict)
    elif priority == 4:  # 4. Order colliders by p_{xz|not y} in descending order
        UC_dict = sort_dict_ascending(UC_dict, True)
        print(UC_dict)

    for (x, y, z) in UC_dict.keys():
        print((x,y,z))
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                 background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue

        if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
            edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
            if edge1 is not None:
                cg_new.G.remove_edge(edge1)
            # Orient only if the edges have not been oriented the other way around
            cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
            print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')

            edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
            if edge2 is not None:
                cg_new.G.remove_edge(edge2)
            cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
            print(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')

            cg_new.definite_UC.append((x, y, z))

    return cg_new

## Define function to calculate shapley values
def shapley(x:int, y:int, cg_1:CausalGraph, verbose:bool=False, ikb: bool = False) -> list():
    num_of_nodes = len(cg_1.G.nodes)
    ## Extract conditioning sets and p-values from the skeleton
    if ikb:
        s_p = [(t.S, t.p_val) for t in cg_1.IKB_list if t.X=={x} and t.Y=={y}]
    else:
        s_p = cg_1.IKB[x,y]
    max_set_size = num_of_nodes - 2
    n_factorial = math.factorial(max_set_size)

    sv_list = []
    for i in range(num_of_nodes):
        sv_is = []
        if i not in {x, y}:
            ## calculate marginal contribution of i
            without_i = [t for t in s_p if i not in t[0]]
            ## select p-values
            s_p_i = [t for t in s_p if i in t[0]]
            if len(s_p_i) == 0:
                ## No conditioning sets contain i
                sv_list.append((i, np.nan))
                continue

            ## calculate shapley value
            for t in s_p_i:
                for s in without_i:
                    if set(s[0]).issubset(set(t[0])) and set(t[0]) - set(s[0]) == {i}: # if s is the only difference between t and t-{i}
                        v_i = t[1] - s[1]
                        w_i = math.factorial(len(s[0])) * math.factorial(max_set_size - len(s[0]) - 1) / n_factorial
                        sv_i = v_i * w_i
                        sv_is.append(sv_i)
                        if verbose:
                            print((t[0],s[0]), sv_i)
            avg_sv_i = sum(sv_is)
            if verbose:
                print("SV of {} = {}".format(i, avg_sv_i))
            sv_list.append((i, avg_sv_i))

    return sv_list

def shapley_cs(cg_new: CausalGraph, priority: int = 2, background_knowledge: BackgroundKnowledge = None, 
                verbose: bool = False, ikb: bool = False, selection: str = 'top') -> CausalGraph:
    """
    Run (ShapPC) to orient unshielded colliders

    Parameters
    ----------
    cg : a CausalGraph object
    priority : rule of resolving conflicts between unshielded colliders (default = 3)
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    """

    assert priority in [0, 1, 2, 3, 4]

    # cg_new = deepcopy(cg)
    # UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        if verbose:
            print((x, y, z))
            print(f"(X{x+1},X{y+1},X{z+1})")
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                 background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue

        cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
        cond_with_y_p = [(S,cg_new.ci_test(x, z, S)) for S in cond_with_y]
        if verbose:
            print(f"cond_with_y: {cond_with_y_p}")
        # ikb_dict_x_z = {k: v for d in cg_new.IKB[x,z] for k, v in d.items()}
        ## Add additional test to IKB_list if not already there
        [append_value(cg_new.IKB, x, z, (S,p)) for S,p in cond_with_y_p]
        if ikb:       
            cg_new.IKB_list = cg_new.IKB_list + \
                [test_obj(X={x},S=set(S),Y={z},p_val=cg_new.ci_test(x, z, S),alpha=cg_new.alpha) for S in cond_with_y if S not in \
                    [tuple(test.S) for test in cg_new.IKB_list if test.X=={x} and test.Y=={z}]]
        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)
        cond_without_y_p = [(S,cg_new.ci_test(x, z, S)) for S in cond_without_y]
        if verbose:
            print(f"cond_without_y:{cond_without_y_p}")
        ## Add additional test to IKB_list if not already there
        [append_value(cg_new.IKB, x, z, (S,p)) for S,p in cond_without_y_p]
        if ikb:
            cg_new.IKB_list = cg_new.IKB_list + \
                [test_obj(X={x},S=set(S),Y={z},p_val=cg_new.ci_test(x, z, S),alpha=cg_new.alpha) for S in cond_without_y if S not in 
                    [tuple(test.S) for test in cg_new.IKB_list if test.X=={x} and test.Y=={z}]]

        if verbose:
            print((x,y,z))
            print(f"(X{x+1},X{y+1},X{z+1})")
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                    background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                    background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                    background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue

        sv_list = shapley(x, z, cg_new)
        ## remove nan values (no conditioning sets contain i)
        sv_list = [sv for sv in sv_list if not np.isnan(sv[1])]
        ## sort shapley values
        sv_list.sort(key=lambda x: x[1], reverse=True)
        if verbose:
            print(sv_list)

        ## Option 1: only take the highest shapley value as a v-structure candidate, if it is not negative
        if selection == 'top':
            ## select the top candidate for dependency (the lowest contribution to the p-value for rejecting the null) 
            y_star = sv_list[0][0]
            if verbose:
                print(y_star)
            if y != y_star or sv_list[0][1] < 0:
                continue
        
        if selection == 'bottom':
            ## select the top candidate for dependency (the lowest contribution to the p-value for rejecting the null) 
            y_star = sv_list[-1][0]
            if verbose:
                print(y_star)
            if y != y_star or sv_list[-1][1] > 0:
                continue

        ## Option 2: accept all the candidates that are in the 2 highest SVs
        elif selection == 'top2':
            if len(sv_list) >= 2:
                if y not in [s[0] for s in sv_list[0:2] if s[1]>0]:
                    continue
            elif len(sv_list) == 1:
                if y not in [s[0] for s in sv_list if s[1]>0]:
                    continue
            else:
                continue            

        ## Option 3: take the highest shapley value as a v-structure candidate if it is higher than the median
        elif selection == 'median':
            median_sv = statistics.median([sv[1] for sv in sv_list])
            if [sv[1] for sv in sv_list if sv[0]==y][0] < median_sv:
                continue

        ## Option 4: accept all the candidates that are higher than the one with the biggest increment
        elif selection == 'top_change':
            arr = pd.DataFrame([sv for sv in sv_list], columns=['Var', 'SV']).sort_values('SV', ascending=True)
            arr['change'] = arr['SV'].diff()[arr['SV']>0]
            arr = arr[arr['SV']>0]
            if len(arr) == 0:
                continue # no positive SVs, all contribute negatively to the p-value
            if y not in arr.loc[np.where(max(arr['change']))[0][0]:,'Var'].values:
                continue # y is not in the top change        

        else:
            raise ValueError(f"Selection method {selection} not recognized")

        if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
            if verbose:
                print(f"{y} -- {x} and {y} -- {z}")

            ### Conclusions: Orient V-structure               
            edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
            if edge1 is not None:
                cg_new.G.remove_edge(edge1, False)
            cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
            # Conclusion1 = "arrow({}, {})".format(x, y)

            if verbose:
                print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                print("Is DAG?", is_dag(cg_new.G.graph > 0))
            
            ##TODO: or we can check which one has the lowest p-value and remove the other one
            
            edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
            if edge2 is not None:
                cg_new.G.remove_edge(edge2, False)
            cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
            if verbose:
                print(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                print("Is DAG?", is_dag(cg_new.G.graph > 0))
            # Conclusion2 = "arrow({}, {})".format(z, y)
            # if maxp_rule:
            #     cg_new.decisions[(UC_premise_test, Premise)].add((Conclusion1, Conclusion2))

            if priority == 0: ## 0. Overwrite
                if not is_dag(cg_new.G.graph > 0):
                    print("Not DAG - Priority 0 to be implemented")
            if priority == 1: ## 1. Orient bi-directed
                if not is_dag(cg_new.G.graph > 0):
                    print("Not DAG - Priority 1 to be implemented")
            if priority == 2: ## 2. Prioritize existing colliders
                if not is_dag(cg_new.G.graph > 0):
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                    edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1, False)
                    if edge2 is not None:
                        cg_new.G.remove_edge(edge2, False)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.TAIL))
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.TAIL))
                    # if verbose:
                    print(f'Removed: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                    print(f'Removed: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                    print("Is DAG?", is_dag(cg_new.G.graph > 0))
            elif priority == 3: ## 3. Prioritize stronger colliders
                if not is_dag(cg_new.G.graph > 0):
                    print("Not DAG - Priority 3 to be implemented")


        # else:
        #     #remove from decisions if edge is not removed
        #     if maxp_rule:
        #         del cg_new.decisions[get_keys_from_value(cg_new.decisions, (UC_premise_test, Premise))]

    return cg_new


def shapley_cs_full(cg: CausalGraph, priority: int = 3, background_knowledge: BackgroundKnowledge = None, verbose: bool = False, ikb: bool = False) -> CausalGraph:
    """
    Run (MaxP) to orient unshielded colliders

    Parameters
    ----------
    cg : a CausalGraph object
    priority : rule of resolving conflicts between unshielded colliders (default = 3)
           0: overwrite
           1: orient bi-directed
           2. prioritize existing colliders
           3. prioritize stronger colliders
           4. prioritize stronger* colliers
    background_knowledge : artificial background background_knowledge

    Returns
    -------
    cg_new : a CausalGraph object. Where cg_new.G.graph[j,i]=1 and cg_new.G.graph[i,j]=-1 indicates  i --> j ,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = -1 indicates i --- j,
                    cg_new.G.graph[i,j] = cg_new.G.graph[j,i] = 1 indicates i <-> j.
    """

    assert priority in [0, 1, 2, 3, 4]

    cg_new = deepcopy(cg)
    UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        if verbose:
            print((x, y, z))
            print(f"(X{x+1},X{y+1},X{z+1})")
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                 background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue

        cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
        cond_with_y_p = [(S,cg_new.ci_test(x, z, S)) for S in cond_with_y]
        if verbose:
            print(f"cond_with_y: {cond_with_y_p}")
        # ikb_dict_x_z = {k: v for d in cg_new.IKB[x,z] for k, v in d.items()}
        ## Add additional test to IKB_list if not already there
        [append_value(cg.IKB, x, z, (S,p)) for S,p in cond_with_y_p]
        if ikb:       
            cg_new.IKB_list = cg_new.IKB_list + \
                [test_obj(X={x},S=set(S),Y={z},p_val=cg_new.ci_test(x, z, S),alpha=cg_new.alpha) for S in cond_with_y if S not in \
                    [tuple(test.S) for test in cg_new.IKB_list if test.X=={x} and test.Y=={z}]]
        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)
        cond_without_y_p = {S:cg_new.ci_test(x, z, S) for S in cond_without_y}
        if verbose:
            print(f"cond_without_y:{cond_without_y_p}")
        ## Add additional test to IKB_list if not already there
        [append_value(cg.IKB, x, z, (S,p)) for S,p in cond_with_y_p]
        if ikb:
            cg_new.IKB_list = cg_new.IKB_list + \
                [test_obj(X={x},S=set(S),Y={z},p_val=cg_new.ci_test(x, z, S),alpha=cg_new.alpha) for S in cond_without_y if S not in 
                    [tuple(test.S) for test in cg_new.IKB_list if test.X=={x} and test.Y=={z}]]

        max_p_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_with_y])
        max_p_not_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_without_y])

        if max_p_not_contain_y > max_p_contain_y:
            if verbose:
                print("max_p_not_contain_y > max_p_contain_y",max_p_not_contain_y, max_p_contain_y)

            if priority == 0:  # 0: overwrite
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                edge2 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[x])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                # Fully orient the edge irrespective of what have been oriented
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')

                edge3 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[z])
                if edge3 is not None:
                    cg_new.G.remove_edge(edge3)
                edge4 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge4 is not None:
                    cg_new.G.remove_edge(edge4)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    print(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')

            elif priority == 1:  # 1: orient bi-directed
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    if cg_new.G.graph[x, y] == Endpoint.TAIL.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[x, y] == Endpoint.ARROW.value and cg_new.G.graph[y, x] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge1)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    if cg_new.G.graph[z, y] == Endpoint.TAIL.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    elif cg_new.G.graph[z, y] == Endpoint.ARROW.value and cg_new.G.graph[y, z] == Endpoint.TAIL.value:
                        cg_new.G.remove_edge(edge2)
                        cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.ARROW, Endpoint.ARROW))
                else:
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))

            elif priority == 2:  # 2: prioritize existing
                if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                    edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                    if edge1 is not None:
                        cg_new.G.remove_edge(edge1)
                    # Orient only if the edges have not been oriented the other way around
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    if verbose:
                        print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')

                    edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                    if edge2 is not None:
                        cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    if verbose:
                        print(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')

            elif priority == 3: # 3: prioritize stronger
                UC_dict[(x, y, z)] = max_p_contain_y
                if verbose:
                    print(f"Chosen: {[{S,cg_new.ci_test(x, z, S)} for S in cond_with_y if cg_new.ci_test(x, z, S)==max_p_contain_y]}")

            elif priority == 4: # 4: prioritize stronger*
                UC_dict[(x, y, z)] = max_p_not_contain_y
                if verbose:
                    print(f"Chosen: {[{S,cg_new.ci_test(x, z, S)} for S in cond_without_y if cg_new.ci_test(x, z, S)==max_p_not_contain_y]}")

                ## Rule based on MaxP. remove for now
                maxp_rule = False
                if maxp_rule == True:
                    loosing_cond_set = get_keys_from_value(cond_with_y_p, max_p_contain_y)
                    winning_cond_set = get_keys_from_value(cond_without_y_p, max_p_not_contain_y)
                    # collider = set(loosing_cond_set) - set(winning_cond_set) #TODO: revise for lenght issues

                    ## Premises 
                    # x _||_ z #
                    ### This is the condition for being UT in the first place
                    UC_premise_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                                        test.S==set() and test.dep_type=='I'][0]                
                    # x _||_ z | {W} #
                    ### The strongest test does not contain y, hence independence is attributed to {W} 
                    # and y considered a collider
                    premise_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                                        test.S==set(winning_cond_set) and test.dep_type=='I']
                    if premise_test:
                        Premise = premise_test[0] 
                    else:
                        Premise = test_obj(X={x}, S=set(winning_cond_set), Y={z}, dep_type="I")
                        cg_new.IKB_list.append(Premise)
                    
                    ## Conclusions
                    # x _|/|_ z | y ### y is collider
                    conc1_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                                        test.S=={y} and test.dep_type=='D']
                    if conc1_test:
                        Conclusion1 = conc1_test[0] 
                    else:
                        Conclusion1 = test_obj(X={x}, S={y}, Y={z}, dep_type="D")
                        cg_new.IKB_list.append(Conclusion1)

                    # x _|/|_ z | {W} + y ### Weaker test is not "believed" hence made a dependence 
                    # even though returns 'I' from the data
                    conc2_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                                        test.S==set(loosing_cond_set) and test.dep_type=='D']
                    if conc2_test:
                        Conclusion2 = conc2_test[0] 
                    else:
                        Conclusion2 = test_obj(X={x}, S=set(loosing_cond_set), Y={z}, dep_type="D")
                        cg_new.IKB_list.append(Conclusion2)
                    
                    full_premise = (UC_premise_test, Premise) if UC_premise_test != Premise else tuple([Premise])
                    full_conclusion = (Conclusion1, Conclusion2) if Conclusion1 != Conclusion2 else tuple([Conclusion1])

                    cg_new.decisions[full_premise].add(full_conclusion)
        else:
            maxp_rule = False
            if maxp_rule:
                ##TODO: do I need this?
                if verbose:
                    print("max_p_not_contain_y <= max_p_contain_y",max_p_not_contain_y, max_p_contain_y)
                loosing_cond_set = get_keys_from_value(cond_without_y_p, max_p_not_contain_y)
                winning_cond_set = get_keys_from_value(cond_with_y_p, max_p_contain_y)
                # non_collider = set(loosing_cond_set) - set(winning_cond_set) #TODO: revise for lenght issues

                ## Premises 
                # x _||_ z #
                ### This is the condition for being UT in the first place
                UC_premise_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                                    test.S==set() and test.dep_type=='I'][0]                
                # x _||_ z | {W} + y #
                ### The strongest test contains y, hence independence is attributed to y
                # and y is NOT considered a collider
                premise_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                                    test.S==set(winning_cond_set) and test.dep_type=='I']
                if premise_test:
                    Premise = premise_test[0] 
                else:
                    Premise = test_obj(X={x}, S=set(winning_cond_set), Y={z}, dep_type="I")
                    cg_new.IKB_list.append(Premise)

                ## Conclusions
                # x _||_ z | y ### y is NOT a collider
                conc1_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                                    test.S=={y} and test.dep_type=='I']
                if conc1_test:
                    Conclusion1 = conc1_test[0] 
                else:
                    Conclusion1 = test_obj(X={x}, S={y}, Y={z}, dep_type="I")
                    cg_new.IKB_list.append(Conclusion1)

                conc2_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                                    test.S==set(loosing_cond_set) and test.dep_type=='D']

                # x _|/|_ z | {W} ### Weaker test is not "believed" hence made a dependence 
                # even though returns I solely from the data
                if conc2_test:
                    Conclusion2 = conc2_test[0] 
                else:
                    Conclusion2 = test_obj(X={x}, S=set(loosing_cond_set), Y={z}, dep_type="D")
                    cg_new.IKB_list.append(Conclusion2)

                full_premise = (UC_premise_test, Premise) if UC_premise_test != Premise else tuple([Premise])
                full_conclusion = (Conclusion1, Conclusion2) if Conclusion1 != Conclusion2 else tuple([Conclusion1])

                cg_new.decisions[full_premise].add(full_conclusion)

    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
            UC_dict = sort_dict_ascending(UC_dict)
            if verbose:
                print('UC_dict', UC_dict)
        else:  # 4. Order colliders by p_{xz|not y} in descending order
            UC_dict = sort_dict_ascending(UC_dict, True)
            if verbose:
                print('UC_dict', UC_dict)

        for (x, y, z) in UT:
            if verbose:
                print((x,y,z))
                print(f"(X{x+1},X{y+1},X{z+1})")
            if (background_knowledge is not None) and \
                    (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                     background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
                continue

            sv_list = shapley(x, z, cg_new)
            ## sort shapley values
            sv_list.sort(key=lambda x: x[1], reverse=False)
            if verbose:
                print(sv_list)
            ## select the top candidate for dependency (the lowest contribution to the p-value for rejecting the null) 
            y_star = sv_list[0][0]
            if verbose:
                print(y_star)
            # median_sv = statistics.median([sv[1] for sv in sv_list])

            ## Option 1: only take the lowest shapley value as a v-structure candidate 
            # if y =! y_star:
            #     continue
            if y not in [s[0] for s in sv_list[0:2]]:
                continue

            if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                if verbose:
                    print(f"{y} -- {x} and {y} -- {z}")

                ### Conclusions: Orient V-structure               
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1, False)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')   
                Conclusion1 = "arrow({}, {})".format(x, y)

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2, False)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                if verbose:
                    print(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                Conclusion2 = "arrow({}, {})".format(z, y)
                # if maxp_rule:
                #     cg_new.decisions[(UC_premise_test, Premise)].add((Conclusion1, Conclusion2))

            # else:
            #     #remove from decisions if edge is not removed
            #     if maxp_rule:
            #         del cg_new.decisions[get_keys_from_value(cg_new.decisions, (UC_premise_test, Premise))]

        return cg_new