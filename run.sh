OMP_NUM_THREADS=1 torchrun \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:20746 \
  --nnodes=1 \
  --nproc_per_node=2 \
  train_task_batch_update_mutli_gpu_multi_batch_8task.py