# MultiGPU_tutorial
MultiGPU and Stream tutorial for NVIDIA GPUs  

# How to use?
Just run the `run.sh` script.  
If you can not run the script, follow the codes below.  

``` shell
chmod u+x run.sh
./run.sh
```

---

# Description
This repository is a tutorial for Multi GPU and Stream features of NVIDIA GPUs.  
The kernel we use is GEMM(GEneral Matrix Multiplication).  
The size of the matrix is configurable.  

There are 5 executables that we can build using `make`  
* cpu: GEMM on CPU(single thread)
* single-nostream: GEMM on Single GPU and NO stream used
* single-stream: GEMM on Single GPU and stream used
* multi-nostream: GEMM on Multi GPU and NO stream used
* multi-stream: GEMM on Multi GPU and stream used

The output of the executable shows how the performance changes when using Multi-GPU and Stream.  
