## Mask Implementation of Network Slimming
During pruning, we set those scaling factors in BN layer which correspond to pruned channels to be 0.  
 When training the pruned model, in each iteration, before
we call `optimizer.step()`, we update the gradient of those 0 scaling factors to be 0. This is achieved in `BN_grad_zero` function.
### Pros
- We don't need to introduce channel selection layer which adds to the training time.
- Even if a layer is pruned to zero channels, it won't raise any error. Instead, this layer will simply output an all-0 tensor.
### Cons
- Not easy to compute flops and parameters.