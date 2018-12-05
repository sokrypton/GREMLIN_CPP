# GREMLIN_CPP v1.0

### UPDATE
We now have an exact implemention of this code in tensorflow!
https://github.com/sokrypton/GREMLIN_CPP/blob/master/GREMLIN_TF.ipynb

### Installation
```sh
$ g++ -O3 -std=c++0x -o gremlin_cpp gremlin_cpp.cpp -fopenmp
```
invoke ```-fopenmp``` to allow usage of multiple CPU(s), the code is openmp friendly.
### Usage
Note, openmp uses the system variable ```OMP_NUM_THREADS``` to decide how many threads/CPU(s) to use.
```
$ export OMP_NUM_THREADS=16
$ ./gremlin_cpp -i alignment_file -o results
# ---------------------------------------------------------------------------------------------
#                                GREMLIN_CPP v1.0                                              
# ---------------------------------------------------------------------------------------------
#   -i            input alignment (either one sequence per line or in fasta format)
#   -o            save output to
# ---------------------------------------------------------------------------------------------
#  Optional settings                                                                           
# ---------------------------------------------------------------------------------------------
#   -only_neff    only compute neff (effective num of seqs)      [Default=0]
#   -only_v       only compute v (1body-term)                    [Default=0]
#   -gap_cutoff   remove positions with > X fraction gaps        [Default=0.5]
#   -alphabet     select: [protein|rna|binary]                   [Default=protein]
#   -eff_cutoff   seq id cutoff for downweighting similar seqs   [Default=0.8]
#   -lambda       L2 regularization weight                       [Default=0.01]
#   -mrf_i        load MRF
#   -mrf_o        save MRF
#   -pair_i       load list of residue pairs (one pair per line, index 0)
# ---------------------------------------------------------------------------------------------
#  Minimizer settings                                                                          
# ---------------------------------------------------------------------------------------------
#   -min_type     select: [lbgfs|cg|none]                        [Default=lbfgs]
#   -max_iter     number of iterations                           [Default=100]
# ---------------------------------------------------------------------------------------------
```
### parsing output
```
i j raw apc ii jj
i = index i
j = index j
raw = l2norm(W)
apc = raw - mean(row) * mean(col) / mean(all)
ii = char-position i
jj = char-position j
```

The out MRF contains 21 values for each position (V) and 21 x 21 values for each pair of positions (W).

The order of the values is as follows: "ARNDCQEGHILKMFPSTWYV-" (where "-" is the gap). 

For RNA the order is "AUCG-", with 5 values for V and 5x5 for W.

For Binary the order is "01-", with 3 values for V and 3x3 for W.
