## Trainin data process issue
excessive memory usage when processing large datasets with DataFrame.apply and storing all results in memory. 
- Training data is split into smaller chunks of size train_chunk_size for processing and saving individually.
- Use a generator-based approach instead of apply to avoid creating large intermediate objects.
Process and append data to a list in chunks to minimize memory usage.

Chunk Processing:

Training data is split into smaller chunks of size train_chunk_size for processing and saving individually.
Each chunk is cached in a separate directory (part_1, part_2, etc.) under train_dataset_cache_dir.
Validation Dataset:

The validation dataset is processed and saved as a single file.
Reduced Memory Usage:

Each chunk is processed and cached independently, avoiding excessive memory consumption.
Dynamic Chunk Size:

The train_chunk_size parameter controls the size of each chunk for efficient processing.


Here's the updated implementation to chunk both the train_dataset and val_dataset into smaller parts, process each part separately, merge them into one dataset each (training and validation), sanitize the datasets, and save them as single directories:
---

## CUDA Out of Memory Issue 

Encountered a **CUDA Out of Memory (OOM)** issue, which occurs when GPU does not have enough free memory to allocate additional resources for the training step. Here's how to resolve it:

---

### **Steps to Resolve OOM Error**
1. **Reduce `per_device_train_batch_size`:**
   The batch size is a major contributor to memory usage. Halve the value in the `TrainingArguments`:
   ```python
   per_device_train_batch_size=8,
   per_device_eval_batch_size=8,
   ```
   If necessary, reduce further to `4` or even `2`.

2. **Enable Gradient Accumulation:**
   Use smaller batches and accumulate gradients to simulate a larger effective batch size:
   ```python
   gradient_accumulation_steps=2,  # Simulates doubling the batch size
   ```
   This splits the batch across multiple forward passes, reducing memory demand.

3. **Use Mixed Precision Training:**
   Enable mixed precision training to reduce memory usage:
   ```python
   fp16=True,
   ```
   This works well if your GPU supports Tensor Cores (NVIDIA Volta and later).

4. **Clear GPU Memory:**
   Ensure no other processes are using GPU memory before training:
   - Check running processes:
     ```bash
     nvidia-smi
     ```
   - Kill unwanted processes:
     ```bash
     kill -9 <process_id>
     ```

5. **Restrict GPU Usage:**
   Limit the script to specific GPUs with more available memory:
   ```bash
   CUDA_VISIBLE_DEVICES=1 python your_script.py
   ```

6. **Optimize the Model:**
   - Use model checkpoints with lower memory requirements.
   - Reduce the sequence length if applicable:
     ```python
     max_length=256,  # Or a smaller value based on your dataset
     ```

7. **Set CUDA Environment Variable:**
   Enable dynamic memory management to prevent fragmentation:
   ```bash
   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

8. **Use Multi-GPU Training:**
   Distribute the workload across multiple GPUs if available:
   ```bash
   torchrun --nproc_per_node=2 your_script.py
   ```

---



## Utilise Multi-GPU 

The script does **not explicitly configure multi-GPU training**, but the Hugging Face `Trainer` does support it automatically if multiple GPUs are available. Here's how it works and what you need to ensure for multi-GPU training:

---

### **Multi-GPU Compatibility in the Script**
1. **`TrainingArguments` Settings**:
   - The `Trainer` automatically uses all available GPUs if you're running the script in an environment with multiple GPUs (e.g., a cluster with `torch.distributed` configured or using `CUDA_VISIBLE_DEVICES`).
   - PyTorch's Distributed Data Parallel (DDP) will handle data distribution across GPUs.

2. **Key Arguments**:
   - The script includes **default multi-GPU support**. If you want finer control, you can adjust the `TrainingArguments` by adding:
     ```python
     deepspeed=None,  # Leave unset unless using DeepSpeed
     ddp_find_unused_parameters=False,  # Speeds up multi-GPU when not using all model parameters
     ```
   - You can also specify the number of GPUs via the `CUDA_VISIBLE_DEVICES` environment variable.

3. **Batch Size Adjustment**:
   - The batch size is specified per device:
     ```python
     per_device_train_batch_size=16,
     per_device_eval_batch_size=16,
     ```
     If you're running on multiple GPUs, the effective total batch size becomes:
     \[
     \text{Total Batch Size} = \text{per\_device\_train\_batch\_size} \times \text{Number of GPUs}
     \]

---

### **Testing Multi-GPU Setup**
To confirm multi-GPU usage, you can check during runtime:
```python
import torch

print(f"Number of GPUs available: {torch.cuda.device_count()}")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
```

---

### **Ensuring Multi-GPU Support**
1. **Launch the Script Correctly**:
   Use the `torchrun` utility for launching the script in a distributed manner:
   ```bash
   torchrun --nproc_per_node=<num_gpus> your_script.py
   ```
   Replace `<num_gpus>` with the number of GPUs you'd like to use.

2. **Environment Variable**:
   Set `CUDA_VISIBLE_DEVICES` to restrict or specify GPUs:
   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 python your_script.py
   ```

3. **Logs**:
   Hugging Face Trainer logs will indicate whether it is using multiple GPUs. Look for entries like `Distributed Training` in the logs.

---

### **If Explicit Multi-GPU Configuration is Needed**
You can add the `torch.distributed` backend explicitly by modifying the script:
```python
training_args = TrainingArguments(
    ...
    local_rank=-1,  # Use -1 for non-distributed; will be set automatically by torchrun
    dataloader_num_workers=4,  # Speed up loading data on multiple GPUs
    distributed_strategy="ddp",  # Explicitly use Distributed Data Parallel
    ...
)
```

