# Deal inconsistency in the S calculation: Revise the definition of k-cut value
works. for 3by3, max gain ratio ~ 40%, while random guess44%/greedy 55%/optimal 56%
                 max gain ratio ~ 32%, while random guess27%/greedy 41%/optimal 42%

running validation benchmark
Reward shaping.
why behave like greedy? maybe we should not have negative reward,
and the reward close to the optimal should be amplified.
what exactly is the influence of gamma. Seems like gamma=1 is ok. 
potential function V(S) to help to reshape R. why the agent always underestimate the Q?

Action pruning.

State-action encoding.
 

1. flip mode
2. m = [2,3,4]
3. rotate graph when testing
4. peek reward. sanity check: feed immediate reward, and learn the greedy. 
5. reconstruct buffer, save s1, allow q-step/TD(lambda)
6. alter gnn structure for generalization of size
7. small \gamma is better, why?
8. greedy benchmark for clustered graph


10 by 10 
--complete
Avg value of initial S: 234.7038572692871/233.924688/235.765625
Avg episode best value: 77.02441688537597/82.699629/82.030137
Avg step cost: 56.38
Avg percentage max gain: 0.6718229611497091/0.6464/0.6521
--incomplete
Avg value of initial S: 17.95418667793274
Avg episode best value: 2.1257452952861784
Avg step cost: 98.96
Avg percentage max gain: 0.8816016936094965

5 by 6 
--complete
Avg value of initial S: 39.173927841186526/38.844648/39.56/39.024864
Avg episode best value: 18.204591732025147/19.607192/19.37/18.484279
Avg step cost: 11.88
Avg percentage max gain: 0.5352880669554592/0.4952/0.5102/0.5263
--incomplete
Avg value of initial S: 6.979210567474365
Avg episode best value: 1.8567707288265227
Avg step cost: 27.59
Avg percentage max gain: 0.7339569123362257

3by3
--complete
Avg value of initial S: 4.7138877820968625/4.72
Avg episode best value: 2.8001146948337556/2.90
Avg step cost: 2.11                                                                                                                                                          
Avg percentage max gain: 0.40598613622741136     
--incomplete
Avg value of  initial S: 2.17603048130318
Avg episode best value: 1.0029113071936149
Avg step cost: 2.197530864197531
Avg percentage max gain: 0.5391097156906589
 



Greedy algorithm
Running time
soft-dqn(end)
soft-dqn(best)
Running time
3 by 3
40.5%
0.42
24.9%
40.3%
0.13
5 by 6
53.5%
13.6
52.4%
54.6%
0.18
10 by 10
67.2%
726.0
65.3%
66.8%
0.35


# 10by10 check nan. 
caused by numerical error in square_dist_matrix. Apply relu clip to avoid it. 
Also be aware that a large density of nodes could be dangerous.

# Fit immediate reward (hold)
# test gnn structure



# test exploration strategy
    1. for different state, change eps
    2. sample more 'difficult' states from buffer
    3. prioritized buffer
    
    There are UCB/Thompson sampling for RL. need to fix training data.

# multi-gpu parallelization


Build a general greedy benchmark 
Reward engineering no gain

python -m torch.utils.bottleneck run.py --gpu=0 --n_epoch=100
--------------------------------------------------------------------------------
  Environment Summary
--------------------------------------------------------------------------------
PyTorch 1.3.1 compiled w/ CUDA 10.0.130
Running with Python 3.7 and 

`pip list` truncated output:
numpy==1.17.2
numpydoc==0.9.1
torch==1.3.1
torchvision==0.4.2
--------------------------------------------------------------------------------
  cProfile output
--------------------------------------------------------------------------------
         151596154 function calls (147005939 primitive calls) in 169.740 seconds

   Ordered by: internal time
   List reduced from 3215 to 15 due to restriction <15>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
423079/15105    6.282    0.000   25.297    0.002 /u/fy4bc/anaconda3/lib/python3.7/copy.py:268(_reconstruct)
   125303    5.414    0.000    5.414    0.000 {method 'nonzero' of 'torch._C._TensorBase' objects}
   183301    5.110    0.000    5.110    0.000 {built-in method cat}
   358800    4.886    0.000    4.886    0.000 {built-in method index_select}
    73040    4.693    0.000    4.693    0.000 {method 'cuda' of 'torch._C._TensorBase' objects}
    15600    4.372    0.000    9.980    0.001 /u/fy4bc/anaconda3/lib/python3.7/site-packages/dgl/runtime/degree_bucketing.py:82(_degree_bucketing_schedule)
   349100    4.240    0.000    4.240    0.000 {method 'item' of 'torch._C._TensorBase' objects}
     5000    4.237    0.001   12.568    0.003 /u/fy4bc/code/research/RL4CombOptm/gnn_rl/k_cut.py:394(<listcomp>)
2920207/20105    3.991    0.000   25.853    0.001 /u/fy4bc/anaconda3/lib/python3.7/copy.py:132(deepcopy)
  4480134    3.736    0.000   11.636    0.000 /u/fy4bc/anaconda3/lib/python3.7/site-packages/dgl/frame.py:864(is_span_whole_column)
   101402    3.642    0.000    5.596    0.000 /u/fy4bc/anaconda3/lib/python3.7/site-packages/dgl/graph.py:899(__init__)
    15600    3.222    0.000    6.357    0.000 /u/fy4bc/code/research/RL4CombOptm/gnn_rl/k_cut.py:420(udf_u_mul_e)
  3652325    2.815    0.000   17.288    0.000 /u/fy4bc/anaconda3/lib/python3.7/site-packages/dgl/frame.py:582(__getitem__)
  3558725    2.789    0.000   13.272    0.000 /u/fy4bc/anaconda3/lib/python3.7/site-packages/dgl/frame.py:619(select_column)
9400274/9023648    2.447    0.000    5.579    0.000 {built-in method builtins.len}


--------------------------------------------------------------------------------
  autograd profiler output (CPU mode)
--------------------------------------------------------------------------------
        top 15 events sorted by cpu_time_total

--------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------  
Name                  Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     CUDA total %     CUDA total       CUDA time avg    Number of Calls  Input Shapes                         
--------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------  
cat                   12.99%           19.103ms         12.99%           19.103ms         19.103ms         NaN              0.000us          0.000us          1                []                                   
split_with_sizes      6.88%            10.111ms         6.88%            10.111ms         10.111ms         NaN              0.000us          0.000us          1                []                                   
sort                  6.44%            9.466ms          6.44%            9.466ms          9.466ms          NaN              0.000us          0.000us          1                []                                   
matmul                6.33%            9.311ms          6.33%            9.311ms          9.311ms          NaN              0.000us          0.000us          1                []                                   
index_select          6.29%            9.252ms          6.29%            9.252ms          9.252ms          NaN              0.000us          0.000us          1                []                                   
bmm                   6.25%            9.197ms          6.25%            9.197ms          9.197ms          NaN              0.000us          0.000us          1                []                                   
add                   6.17%            9.068ms          6.17%            9.068ms          9.068ms          NaN              0.000us          0.000us          1                []                                   
index_select          6.15%            9.045ms          6.15%            9.045ms          9.045ms          NaN              0.000us          0.000us          1                []                                   
nonzero               6.14%            9.029ms          6.14%            9.029ms          9.029ms          NaN              0.000us          0.000us          1                []                                   
index_select          6.12%            9.001ms          6.12%            9.001ms          9.001ms          NaN              0.000us          0.000us          1                []                                   
add                   6.07%            8.930ms          6.07%            8.930ms          8.930ms          NaN              0.000us          0.000us          1                []                                   
index_select          6.05%            8.890ms          6.05%            8.890ms          8.890ms          NaN              0.000us          0.000us          1                []                                   
sort                  6.04%            8.884ms          6.04%            8.884ms          8.884ms          NaN              0.000us          0.000us          1                []                                   
to                    6.04%            8.884ms          6.04%            8.884ms          8.884ms          NaN              0.000us          0.000us          1                []                                   
index_select          6.03%            8.872ms          6.03%            8.872ms          8.872ms          NaN              0.000us          0.000us          1                []                                   
--------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------  
Self CPU time total: 147.043ms
CUDA time total: 0.000us

--------------------------------------------------------------------------------
  autograd profiler output (CUDA mode)
--------------------------------------------------------------------------------
        top 15 events sorted by cpu_time_total

	Because the autograd profiler uses the CUDA event API,
	the CUDA time column reports approximately max(cuda_time, cpu_time).
	Please ignore this output if your code does not use CUDA.

-----------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------  
Name                     Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     CUDA total %     CUDA total       CUDA time avg    Number of Calls  Input Shapes                         
-----------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------  
set_                     13.16%           39.422ms         13.16%           39.422ms         39.422ms         13.29%           39.424ms         39.424ms         1                []                                   
item                     7.45%            22.313ms         7.45%            22.313ms         22.313ms         7.52%            22.312ms         22.312ms         1                []                                   
_local_scalar_dense      7.44%            22.287ms         7.44%            22.287ms         22.287ms         7.51%            22.296ms         22.296ms         1                []                                   
topk                     6.92%            20.745ms         6.92%            20.745ms         20.745ms         6.99%            20.752ms         20.752ms         1                []                                   
topk                     6.67%            19.998ms         6.67%            19.998ms         19.998ms         6.74%            20.000ms         20.000ms         1                []                                   
topk                     6.58%            19.727ms         6.58%            19.727ms         19.727ms         6.65%            19.744ms         19.744ms         1                []                                   
index                    6.56%            19.662ms         6.56%            19.662ms         19.662ms         6.62%            19.648ms         19.648ms         1                []                                   
index_select             6.42%            19.226ms         6.42%            19.226ms         19.226ms         6.48%            19.224ms         19.224ms         1                []                                   
item                     6.41%            19.206ms         6.41%            19.206ms         19.206ms         6.47%            19.200ms         19.200ms         1                []                                   
_local_scalar_dense      6.41%            19.193ms         6.41%            19.193ms         19.193ms         6.47%            19.192ms         19.192ms         1                []                                   
item                     5.51%            16.507ms         5.51%            16.507ms         16.507ms         5.57%            16.512ms         16.512ms         1                []                                   
_local_scalar_dense      5.50%            16.488ms         5.50%            16.488ms         16.488ms         5.56%            16.496ms         16.496ms         1                []                                   
topk                     5.20%            15.568ms         5.20%            15.568ms         15.568ms         5.25%            15.568ms         15.568ms         1                []                                   
item                     4.94%            14.799ms         4.94%            14.799ms         14.799ms         4.99%            14.800ms         14.800ms         1                []                                   
select                   4.83%            14.474ms         4.83%            14.474ms         14.474ms         3.88%            11.520ms         11.520ms         1                []                                   
-----------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------  -----------------------------------  
Self CPU time total: 299.616ms
CUDA time total: 296.688ms
