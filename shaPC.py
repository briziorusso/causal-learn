

## Calculate shapley values for the candidate variables in a conditioning set
#  @param X: the candidate variables
#  @param Y: the target variable
#  @param S: the conditioning set
#  @param alpha: the significance level
#  @param data: the data
#  @param v: the value function
#  @return: the shapley values
#  @note: this is a helper function for shapley()

# from causallearn.utils.PCUtils import Helper, Meek, SkeletonDiscovery, UCSepset
# from tests.utils_simulate_data import simulate_discrete_data, simulate_linear_continuous_data
# from causallearn.utils.cit import *
# import math

# ## Simulate data
# ## Graph specification: Example from Colombo
# truth_DAG_directed_edges = {(0, 2), (0, 3), (0, 4), (2, 4), (2, 3), (3, 4), (1, 2), (1, 4)}
# # ## Graph specification: Double Collider
# # # truth_DAG_directed_edges = {(0, 2), (1, 2), (3, 2)}
# # ## Graph specification: Collider
# # # truth_DAG_directed_edges = {(0, 2), (1, 2)}
# num_of_nodes = max(sum(truth_DAG_directed_edges, ())) + 1
# alpha = 0.1
# # ##TODO: prevent PC from stopping when unconditional independence test is found.

# data = simulate_discrete_data(num_of_nodes, 10000, truth_DAG_directed_edges, 42)
# X = data.astype(np.double)

# ## there is no randomness in data generation (with seed fixed for simulate_data).
# ## however, there still exists randomness in KCI (null_sample_spectral).
# ## for this simple test, we can assume that KCI always returns the correct result (despite randomness).

# ## Run PC with default parameters: stable=True, uc_rule=0 (uc_sepset), uc_priority=2 (prioritize existing colliders)
# # cg = pc(data=data, alpha=0.05, stable=True, uc_rule=0, uc_priority=-1, show_progress=True, indep_test='gsq')
# # Run PC with: stable=True, uc_rule=0 (uc_sepset), uc_priority=0 (overwrite)


# alpha=0.05
# indep_test=fisherz 
# stable: bool = True 
# ikb: bool = False
# uc_rule: int = 0 
# uc_priority: int = 2

# indep_test = CIT(data, indep_test)
# cg_1 = SkeletonDiscovery.skeleton_discovery(data, alpha, indep_test, keep_edges=True)


# ## run shapley calculation for a pair of variables
# x = 1
# y = 3


# ## Define function to calculate shapley values
# def shapley(x:int, y:int, num_of_nodes:int, cg_1:SkeletonDiscovery, verbose:bool=True) -> list():

#     ## Extract conditioning sets and p-values from the skeleton
#     s_p = [(t.S, t.p_val) for t in cg_1.IKB_list if t.X=={x} and t.Y=={y}]
#     max_set_size = num_of_nodes - 2
#     n_factorial = math.factorial(max_set_size)

#     sv_list = []
#     for i in range(num_of_nodes):
#         sv_is = []
#         if i not in {x, y}:
#             ## calculate marginal contribution of i
#             without_i = [t for t in s_p if i not in t[0]]
#             ## select p-values
#             s_p_i = [t for t in s_p if i in t[0]]
#             ## calculate shapley value
#             for t in s_p_i:
#                 for s in without_i:
#                     if set(s[0]).issubset(set(t[0])) and set(t[0]) - set(s[0]) == {i}:
#                         v_i = t[1] - s[1]
#                         w_i = math.factorial(len(s[0])) * math.factorial(max_set_size - len(s[0]) - 1) / n_factorial
#                         sv_i = v_i * w_i
#                         sv_is.append(sv_i)
#                         if verbose:
#                             print((t[0],s[0]), sv_i)
#             avg_sv_i = sum(sv_is)
#             if verbose:
#                 print("SV of {} = {}".format(i, avg_sv_i))
#             sv_list.append((i, avg_sv_i))

#     return sv_list

# sv_list = shapley(x, y, num_of_nodes, cg_1)
# ## sort shapley values
# sv_list.sort(key=lambda x: x[1], reverse=True)
# print(sv_list)
# ## select the top variable
# x_star = sv_list[0][0]
# print(x_star)

# ## is there any other means to chose the size of the conditioning set?

# ## run PC with the selected conditioning set
# S = [x_star]

# """Experiments for linear Gaussian SEM with two variables."""
# import os, sys
# sys.path.append("../")
# from notears import utils
# from notears.nonlinear import notears_nonlinear, NotearsMLP
# # from notears.notears import notears, utils
# import numpy as np
# import networkx as nx
# from tqdm import tqdm
# from collections import defaultdict
# from causallearn.search.ConstraintBased.PC import pc
# from cdt.data import AcyclicGraphGenerator


# def main():
#     utils.set_random_seed(123)

#     num_graph = 10
#     num_data_per_graph = 1

#     n, d, s0, graph_type, sem_type = 100, 5, 1, 'ER', 'gauss'

#     # equal variance
#     w_ranges = ((-2.0, -0.5), (0.5, 2.0))
#     noise_scale = [1., 1., 1., 1., 1.]
#     expt_name = 'equal_var'
#     run_expt(num_graph, num_data_per_graph, n, d, s0, graph_type, sem_type, w_ranges, noise_scale, expt_name)

#     # # large a
#     # w_ranges = ((-2.0, -1.1), (1.1, 2.0))
#     # noise_scale = [1., 0.15]
#     # expt_name = 'large_a'
#     # run_expt(num_graph, num_data_per_graph, n, d, s0, graph_type, sem_type, w_ranges, noise_scale, expt_name)

#     # # small a
#     # w_ranges = ((-0.9, -0.5), (0.5, 0.9))
#     # noise_scale = [1, 0.15]
#     # expt_name = 'small_a'
#     # run_expt(num_graph, num_data_per_graph, n, d, s0, graph_type, sem_type, w_ranges, noise_scale, expt_name)


# def run_expt(num_graph, num_data_per_graph, n, d, s0, graph_type, sem_type, w_ranges, noise_scale, expt_name):
#     # os.mkdir(expt_name)
#     # os.chmod(expt_name, 0o777)
#     perf = defaultdict(list)
#     perf2 = defaultdict(list)
#     for ii in tqdm(range(num_graph)):
#         ##CDT
#         # generator = AcyclicGraphGenerator('linear', npoints=n, nodes=d)
#         # data, graph = generator.generate(rescale=True)
#         # B_true = generator.adjacency_matrix
#         ##NOTEARS
#         # B_true = utils.simulate_dag(d, s0, graph_type)
#         # W_true = utils.simulate_parameter(B_true, w_ranges=w_ranges)
#         B_true = np.array([[0, 0, 1, 1, 1],
#                             [0, 0, 1, 0, 1],
#                             [0, 0, 0, 1, 1],
#                             [0, 0, 0, 0, 1],
#                             [0, 0, 0, 0, 0]])
#         # W_true_fn = os.path.join(expt_name, f'graph{ii:05}_W_true.csv')
#         # np.savetxt(W_true_fn, W_true, delimiter=',')
#         for jj in range(num_data_per_graph):
#             ##NOTEARS
#             # X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale)
#             # ALTERNATIVE: utils.simulate_nonlinear_sem(B_true, n, sem_type)
#             ##CDT
#             # X = data.to_numpy()
#             ##CAUSAL LEARN
#             # X = simulate_discrete_data(d, n, set(nx.DiGraph(B_true).edges), 42)
            
#             # X_fn = os.path.join(expt_name, f'graph{ii:05}_data{jj:05}_X.csv')
#             # np.savetxt(X_fn, X, delimiter=',')
#             # notears
#             model = NotearsMLP(dims=[d, 10, 1], bias=True)
#             W_notears = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)
#             assert utils.is_dag(W_notears)
#             # W_notears_fn = os.path.join(expt_name, f'graph{ii:05}_data{jj:05}_W_notears.csv')
#             # np.savetxt(W_notears_fn, W_notears, delimiter=',')
#             # eval
#             B_notears = (W_notears != 0)
#             acc = utils.count_accuracy(B_true, B_notears)
#             for metric in acc:
#                 perf[metric].append(acc[metric])

#             model2 = pc(data=X, alpha=0.1, ikb=False, uc_rule=3, uc_priority=4, keep_edges=False)
#             W_pc = model2.G.graph
#             assert utils.is_dag(W_pc)
#             # W_notears_fn = os.path.join(expt_name, f'graph{ii:05}_data{jj:05}_W_notears.csv')
#             # np.savetxt(W_notears_fn, W_notears, delimiter=',')
#             # eval
#             B_pc = (W_pc != 0)
#             acc = utils.count_accuracy(B_true, B_pc)
#             for metric in acc:
#                 perf2[metric].append(acc[metric])


#     # print stats
#     print(expt_name, 'notears')
#     for metric in perf:
#         print(metric, f'{np.mean(perf[metric]):.4f}', '+/-', f'{np.std(perf[metric]):.4f}')
#     print(expt_name, 'pc')
#     for metric in perf2:
#         print(metric, f'{np.mean(perf2[metric]):.4f}', '+/-', f'{np.std(perf2[metric]):.4f}')

# if __name__ == '__main__':
#     main()


import numpy as np

def save_pickle(obj, filename, verbose=True):
        dir = os.path.dirname(filename)
        if not os.path.exists(dir):
            os.mkdir(dir)
        with open(filename, 'wb') as file:
            pickle.dump(obj, file, protocol=-1)
        if verbose:
            print(f'Dumped PICKLE to {filename}')

def varsortability(X, W, debug=False):
    """ Takes n x d data and a d x d adjaceny matrix,
    where the i,j-th entry corresponds to the edge weight for i->j,
    and returns a value indicating how well the variance order
    reflects the causal order. """
    E = W != 0
    Ek = E.copy()
    var = np.var(X, axis=0, keepdims=True)
    tol = var.min() * 1e-9

    n_paths = 0
    n_correctly_ordered_paths = 0

#     for k in range(E.shape[0] - 1):
    n_paths += Ek.sum()
    if debug:
        print("Ek.sum",Ek.sum())
        print("n_paths",n_paths)
        print("var",var, var.T, (var / var.T))
        print("n_corrrect", n_correctly_ordered_paths)
    n_correctly_ordered_paths += np.multiply(Ek,var / var.T > 1 + tol).sum()
    if debug:
        print("n_corrrect", n_correctly_ordered_paths,"/",n_paths)
    if debug:
        print("n_correct: varsort", (var / var.T > 1 + tol), "n_var_paths", (var / var.T > 1 + tol).sum(),
              "masking", np.multiply(Ek,var / var.T > 1 + tol),
              "all", n_correctly_ordered_paths, "tol",1 + tol)
        print("Ek",Ek)
        print("E", E)
        Ek = Ek.dot(E)
        print("Ekd", Ek)

    return n_correctly_ordered_paths / n_paths


import os, sys, time, pickle
sys.path.append("../")
from notears import utils
from notears.nonlinear import notears_nonlinear, NotearsMLP
# from notears.notears import notears, utils
from sklearn.preprocessing import StandardScaler  
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from cdt.data import AcyclicGraphGenerator
import networkx as nx
from tests.utils_simulate_data import simulate_discrete_data, simulate_linear_continuous_data
from ananke.graphs import ADMG


num_graph = 10
num_data_per_graph = 1
random_graph = 'toy'
DGP = 'nt'

n, d, s0, graph_type, sem_type = 1000, 5, 10, 'ER', 'gauss' ##low s0 does not work well, TODO: why?

# equal variance
# w_ranges = ((-2.0, -0.5), (0.5, 2.0))
noise_scale = [1., 1., 1., 1., 1.]
expt_name = 'cdt_lin_gauss'
# run_expt(num_graph, num_data_per_graph, n, d, s0, graph_type, sem_type, w_ranges, noise_scale, expt_name)

# # large a
# w_ranges = ((-2.0, -1.1), (1.1, 2.0))
# noise_scale = [1., 0.15]
# expt_name = 'large_a'
# run_expt(num_graph, num_data_per_graph, n, d, s0, graph_type, sem_type, w_ranges, noise_scale, expt_name)

# # small a
w_ranges = ((-0.9, -0.5), (0.5, 0.9))
# noise_scale = [1, 0.15]
# expt_name = 'small_a'
# run_expt(num_graph, num_data_per_graph, n, d, s0, graph_type, sem_type, w_ranges, noise_scale, expt_name)


# def run_expt(num_graph, num_data_per_graph, n, d, s0, graph_type, sem_type, w_ranges, noise_scale, expt_name):
# os.mkdir(expt_name)
# os.chmod(expt_name, 0o777)
perf = defaultdict(list)
perf2 = defaultdict(list)
perf3 = defaultdict(list)
perf4 = defaultdict(list)
for ii in tqdm(range(num_graph)):
    utils.set_random_seed(123+ii)
    if random_graph == 'cdt':
        generator = AcyclicGraphGenerator('linear', npoints=n, nodes=d)
        data, graph = generator.generate(rescale=True)
        B_true = generator.adjacency_matrix
    elif random_graph == 'nt':
        B_true = utils.simulate_dag(d, s0, graph_type)
    else:
        ## Controlled Examples
        # ## Graph specification: Double Collider
        # # truth_DAG_directed_edges = {(0, 2), (1, 2), (3, 2)}
        # ## Graph specification: Collider
        # # truth_DAG_directed_edges = {(0, 2), (1, 2)}
        ## Graph specification: Example from Colombo
        # truth_DAG_directed_edges = {(0, 2), (0, 3), (0, 4), (2, 4), (2, 3), (3, 4), (1, 2), (1, 4)}
        B_true = np.array( [[ 0,  0,  0,  0,  0],
                            [ 0,  0,  0,  0,  0],
                            [ 1,  1,  0,  0,  0],
                            [ 1,  0,  1,  0,  0],
                            [ 1,  1,  1,  1,  0]])
    truth_DAG_directed_edges = set()
    for i in range(B_true.shape[0]):
        for j in range(B_true.shape[1]):
            if B_true[i,j] == 1:
                truth_DAG_directed_edges.add((j,i))
    print([("X{}".format(a+1),"X{}".format(b+1)) for a,b in truth_DAG_directed_edges])
    # G = ADMG([str(n) for n in list(range(d))], [(str(b),str(e)) for b,e in truth_DAG_directed_edges])
    # G.draw(direction="TD")

    for jj in range(num_data_per_graph):
        ##---------NOTEARS DATA
        if DGP == 'nt':
            W_true = utils.simulate_parameter(B_true, w_ranges=w_ranges)
            if sem_type in ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson']:
                X = utils.simulate_linear_sem(W_true, n, sem_type, noise_scale=noise_scale)
            elif sem_type in ['mlp', 'mim', 'gp', 'gp-add']: 
                X = utils.simulate_nonlinear_sem(B_true, n, sem_type)

            # W_true_fn = os.path.join(expt_name, f'graph{ii:05}_W_true.csv')
            # np.savetxt(W_true_fn, W_true, delimiter=',')

        ##---------CDT ## Built together with the DAG in loop before
        elif DGP == 'cdt':
            X = data.to_numpy()

        
        ##---------Causal-learn data
        elif DGP == 'cl_discrete':
            data = simulate_discrete_data(d, n, truth_DAG_directed_edges,  42)
            X = data.astype(np.double).to_numpy()
        elif DGP == 'cl_continuous':
            data = simulate_linear_continuous_data(d, n, truth_DAG_directed_edges, "gaussian", 42)
            X = data.astype(np.double).to_numpy()

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        # X_fn = os.path.join(expt_name, f'graph{ii:05}_data{jj:05}_X.csv')
        # np.savetxt(X_fn, X, delimiter=',')
        print("Varsortability =", varsortability(X, B_true, debug=False))

        ##---------MODELS--------------
        # notears
        start = time.time()
        model = NotearsMLP(dims=[d, 10, 1], bias=True)
        W_notears = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)
        elapsed = time.time() - start
        assert utils.is_dag(W_notears)
        # W_notears_fn = os.path.join(expt_name, f'graph{ii:05}_data{jj:05}_W_notears.csv')
        # np.savetxt(W_notears_fn, W_notears, delimiter=',')
        # eval
        B_notears = (W_notears != 0)
        acc = utils.count_accuracy(B_true, B_notears)
        for metric in acc:
            perf[metric].append(acc[metric])
        perf['time'].append(elapsed)
        perf_fn = os.path.join("outputs",expt_name, f'graph{ii:05}_data{jj:05}_perf_nt.pkl')
        save_pickle(perf, perf_fn)

        model2 = pc(data=X, alpha=0.1, uc_rule=0, uc_priority=3)
        model2.draw_pydot_graph()
        W_pc = model2.G.graph
        print(W_pc)
        # assert utils.is_dag(W_pc)
        # W_notears_fn = os.path.join(expt_name, f'graph{ii:05}_data{jj:05}_W_notears.csv')
        # np.savetxt(W_notears_fn, W_notears, delimiter=',')
        # eval
        B_pc = (W_pc > 0)
        acc = utils.count_accuracy(B_true, B_pc)
        for metric in acc:
            perf2[metric].append(acc[metric])
        perf2['time'].append(model2.PC_elapsed)
        perf_fn = os.path.join("outputs",expt_name, f'graph{ii:05}_data{jj:05}_perf_pc.pkl')
        save_pickle(perf2, perf_fn)

        model3 = pc(data=X, alpha=0.1, ikb=False, uc_rule=3, uc_priority=4, keep_edges=False)
        model3.draw_pydot_graph()
        W_spc = model3.G.graph
        print(W_spc)
        # assert utils.is_dag(W_pc)
        # W_notears_fn = os.path.join(expt_name, f'graph{ii:05}_data{jj:05}_W_notears.csv')
        # np.savetxt(W_notears_fn, W_notears, delimiter=',')
        # eval
        B_spc = (W_spc > 0)
        acc = utils.count_accuracy(B_true, B_spc)
        for metric in acc:
            perf3[metric].append(acc[metric])
        perf3['time'].append(model3.PC_elapsed)
        perf_fn = os.path.join("outputs",expt_name, f'graph{ii:05}_data{jj:05}_perf_spc.pkl')
        save_pickle(perf3, perf_fn)

        start = time.time()
        model4 = ges(X)
        elapsed = time.time() - start
        # model4.draw_pydot_graph()
        W_ges = model4['G'].graph
        print(W_ges)
        # assert utils.is_dag(W_pc)
        # W_notears_fn = os.path.join(expt_name, f'graph{ii:05}_data{jj:05}_W_notears.csv')
        # np.savetxt(W_notears_fn, W_notears, delimiter=',')
        # eval
        B_ges = (W_ges > 0)
        acc = utils.count_accuracy(B_true, B_ges)
        for metric in acc:
            perf4[metric].append(acc[metric]) 
        perf4['time'].append(elapsed)
        ## save results
        perf_fn = os.path.join("outputs",expt_name, f'graph{ii:05}_data{jj:05}_perf_ges.pkl')
        save_pickle(perf4, perf_fn)

# print stats
print(expt_name, 'notears')

for metric in perf:
    print(metric, f'{np.mean(perf[metric]):.4f}', '+/-', f'{np.std(perf[metric]):.4f}')
print(expt_name, 'pc')
for metric in perf2:
    print(metric, f'{np.mean(perf2[metric]):.4f}', '+/-', f'{np.std(perf2[metric]):.4f}')
print(expt_name, 'spc')
for metric in perf3:
    print(metric, f'{np.mean(perf3[metric]):.4f}', '+/-', f'{np.std(perf3[metric]):.4f}')
print(expt_name, 'ges')
for metric in perf4:
    print(metric, f'{np.mean(perf4[metric]):.4f}', '+/-', f'{np.std(perf4[metric]):.4f}')

