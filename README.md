# Artifacts
For "Towards Workload-aware Optimization and Reconfiguration of ML Serving Pipelines"


## vLLM Request Rate Scaling Experiments
The following script benchmarks the latency/throughput of vLLM as we scale up the request rate for different types of parallelism (data, pipeline, tensor). 

```
sh ./request_rate_scaling_experiments.sh <model_name> <num_gpus>
```

## vLLM Hybrid Parallelism Experiments
The following script benchmarks the latency/throughput of vLLM with different combinations of data/pipline/tensor parallelism. Assumes a node with 8 GPUs.

```
sh ./hybrid_parallelism_experiments.sh <model_name> <request_rate>
```