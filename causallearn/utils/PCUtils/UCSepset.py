from __future__ import annotations

from copy import deepcopy

from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GraphClass import CausalGraph
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.Helper import sort_dict_ascending


class test_obj( object ):
    def __init__(self, X:set, S:set, Y:set, p_val:float=None, dep_type:str="I", alpha:float=0.01 ):
        self.X= X
        self.Y= Y
        self.S= S
        self.p_val = p_val
        if p_val == None:
            self.dep_type = dep_type
        else:
            self.dep_type = "D" if p_val < alpha else "I"

    def to_list(self)->list:
        return [self.X, self.S, self.Y, self.dep_type]

    def elements(self):
        return (self.X, self.S, self.Y)
        
    def negate(self):
        if self.dep_type=="D":
            self.dep_type="I"
        else:
            self.dep_type="D"
        return self

    ## Symmetry
    def symmetrise(self, verbose=True):
        if self.dep_type=="I":
            return test_obj(X=self.Y, S=self.S, Y=self.X, p_val=self.p_val, dep_type=self.dep_type)

    ## Decomoposition
    def decompose(self)->dict:
        assert len(self.Y)==2
        Y, W = self.Y
        return {'P':[self.to_list(),self.to_list()], 'C':[[self.X,self.S,Y, self.dep_type],[self.dep_type, self.X,self.S,W, self.dep_type]]}

    ## Weak Union
    def weak_union(self)->dict:
        assert len(self.Y)>=2
        self.Y, self.W = self.Y
        return {'P':self.to_list(), 'C':[self.X, self.S.union({self.W}), self.Y, self.dep_type]}

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val][0]

def get_keys_from_list_of_values(d, val):
    try:
        return [k for k, v in d.items() for subv in v if val in subv][0]
    except:
        return ''

def uc_sepset(cg: CausalGraph, priority: int = 3,
              background_knowledge: BackgroundKnowledge | None = None) -> CausalGraph:
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

    cg_new = deepcopy(cg)

    R0 = []  # Records of possible orientations
    UC_dict = {}
    UT = [(i, j, k) for (i, j, k) in cg_new.find_unshielded_triples() if i < k]  # Not considering symmetric triples

    for (x, y, z) in UT:
        print((x,y,z))
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                 background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue
        if all(y not in S for S in cg.sepset[x, z]): ##NOTE: this is CPC, double check in pcalg
            print(f"all(y={y} not in S for S in cg.sepset[{x}, {z}]={cg.sepset[x, z]})")
            
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
                    print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                    edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                    if edge2 is not None:
                        cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
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

        return cg_new


def maxp(cg: CausalGraph, priority: int = 3, background_knowledge: BackgroundKnowledge = None):
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
        print((x, y, z))
        if (background_knowledge is not None) and \
                (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                 background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                 background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
            continue

        cond_with_y = cg_new.find_cond_sets_with_mid(x, z, y)
        cond_with_y_p = {S:cg_new.ci_test(x, z, S) for S in cond_with_y}
        print(f"cond_with_y: {cond_with_y_p}")
        # ikb_dict_x_z = {k: v for d in cg_new.IKB[x,z] for k, v in d.items()}
        cg_new.IKB_list = cg_new.IKB_list + [test_obj(X={x},S=set(S),Y={z},p_val=cg_new.ci_test(x, z, S),alpha=0.05) for S in cond_with_y if S not in [tuple(test.S) for test in cg_new.IKB_list if test.X=={x} and test.Y=={z}]]
        cond_without_y = cg_new.find_cond_sets_without_mid(x, z, y)
        cond_without_y_p = {S:cg_new.ci_test(x, z, S) for S in cond_without_y}
        print(f"cond_without_y:{cond_without_y_p}")
        cg_new.IKB_list = cg_new.IKB_list + [test_obj(X={x},S=set(S),Y={z},p_val=cg_new.ci_test(x, z, S),alpha=0.05) for S in cond_without_y if S not in [tuple(test.S) for test in cg_new.IKB_list if test.X=={x} and test.Y=={z}]]

        max_p_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_with_y])
        max_p_not_contain_y = max([cg_new.ci_test(x, z, S) for S in cond_without_y])

        if max_p_not_contain_y > max_p_contain_y:
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
                print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')

                edge3 = cg_new.G.get_edge(cg_new.G.nodes[y], cg_new.G.nodes[z])
                if edge3 is not None:
                    cg_new.G.remove_edge(edge3)
                edge4 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge4 is not None:
                    cg_new.G.remove_edge(edge4)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
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
                    print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')

                    edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                    if edge2 is not None:
                        cg_new.G.remove_edge(edge2)
                    cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                    print(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')

            elif priority == 3:
                UC_dict[(x, y, z)] = max_p_contain_y
                print(f"Chosen: {[{S,cg_new.ci_test(x, z, S)} for S in cond_with_y if cg_new.ci_test(x, z, S)==max_p_contain_y]}")

            elif priority == 4:
                UC_dict[(x, y, z)] = max_p_not_contain_y
                print(f"Chosen: {[{S,cg_new.ci_test(x, z, S)} for S in cond_without_y if cg_new.ci_test(x, z, S)==max_p_not_contain_y]}")

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
                # even though returns I solely from the data
                conc2_test = [test for test in cg_new.IKB_list if test.X=={x} and test.Y=={z} and \
                                    test.S==set(loosing_cond_set) and test.dep_type=='D']
                if conc2_test:
                    Conclusion2 = conc2_test[0] 
                else:
                    Conclusion2 = test_obj(X={x}, S=set(loosing_cond_set), Y={z}, dep_type="D")
                    cg_new.IKB_list.append(Conclusion2)

                cg_new.decisions[(UC_premise_test, Premise)].append(set([Conclusion1, Conclusion2]))
        else:
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

            cg_new.decisions[(UC_premise_test,Premise)].append(set([Conclusion1, Conclusion2]))

    if priority in [0, 1, 2]:
        return cg_new

    else:
        if priority == 3:  # 3. Order colliders by p_{xz|y} in ascending order
            UC_dict = sort_dict_ascending(UC_dict)
            print('UC_dict', UC_dict)
        else:  # 4. Order colliders by p_{xz|not y} in descending order
            UC_dict = sort_dict_ascending(UC_dict, True)
            print('UC_dict', UC_dict)

        for (x, y, z) in UC_dict.keys():
            print((x,y,z))
            if (background_knowledge is not None) and \
                    (background_knowledge.is_forbidden(cg_new.G.nodes[x], cg_new.G.nodes[y]) or
                     background_knowledge.is_forbidden(cg_new.G.nodes[z], cg_new.G.nodes[y]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[x]) or
                     background_knowledge.is_required(cg_new.G.nodes[y], cg_new.G.nodes[z])):
                continue

            if (not cg_new.is_fully_directed(y, x)) and (not cg_new.is_fully_directed(y, z)):
                # Orient only if the edges have not been oriented the other way around
                ##NOTE shall I code this condition too?
                print(f"{y} -- {x} and {y} -- {z}")

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
                    Premise = test_obj(X={x}, S={y}, Y={z}, dep_type="D")
                    cg_new.IKB_list.append(Premise)  

                ### Conclusions: Orient V-structure               
                edge1 = cg_new.G.get_edge(cg_new.G.nodes[x], cg_new.G.nodes[y])
                if edge1 is not None:
                    cg_new.G.remove_edge(edge1)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[x], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                print(f'Oriented: x={x} --> y={y} ({cg_new.G.nodes[x].get_name()} --> {cg_new.G.nodes[y].get_name()})')   
                Conclusion = ("orient", (x, y))
                cg_new.decisions[(UC_premise_test,Premise)].append(Conclusion)

                edge2 = cg_new.G.get_edge(cg_new.G.nodes[z], cg_new.G.nodes[y])
                if edge2 is not None:
                    cg_new.G.remove_edge(edge2)
                cg_new.G.add_edge(Edge(cg_new.G.nodes[z], cg_new.G.nodes[y], Endpoint.TAIL, Endpoint.ARROW))
                print(f'Oriented: z={z} --> y={y} ({cg_new.G.nodes[z].get_name()} --> {cg_new.G.nodes[y].get_name()})')
                Conclusion = ("orient", (z, y))
                cg_new.decisions[(UC_premise_test,Premise)].append(Conclusion)

            else:
                #remove from decisions if edge is not removed
                del cg_new.decisions[get_keys_from_value(cg_new.decisions, test_obj(X={x}, S=y, Y={z}, dep_type="D"))]

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
