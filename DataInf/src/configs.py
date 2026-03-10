"""Experiment configuration presets for GLUE mislabeled-data runs."""

import copy
import getpass
import socket

def _setup_env():
    """Capture lightweight runtime metadata for the current machine/user."""
    return {
        'host': socket.gethostname(),
        'user': getpass.getuser(),
    }

def generate_config(expno_name='mrpc',
                    task='mrpc',
                     model='llama',
                     low_rank=2,
                     n_runs=3):    
    """Build the experiment header plus the repeated run configurations."""
    
    # Experiment configuration
    exp = _setup_env()
    exp['expno']=expno_name
    exp['n_runs']=n_runs

    # Run configuration
    run_temp = dict()
    run_temp['task']=task
    run_temp['model']=model
    run_temp['noise_ratio'] = 0.2  # Flip 20% of training labels to synthesize mislabeled samples.
    run_temp['device'] = "cuda"  # All provided experiments expect a CUDA-enabled GPU.
    run_temp['lr'] = 3e-4  # Shared LoRA fine-tuning learning rate for GLUE runs.
    
    run_temp['model_name_or_path'] = "roberta-large"  # Sequence-classification backbone used in the paper demo.
    run_temp['batch_size'] = 32  # Shared mini-batch size for both train and validation loaders.
    run_temp['num_epochs'] = 10  # Fine-tune for a fixed number of passes over the sampled GLUE subset.
    run_temp['target_modules'] = ["value"]  # Apply LoRA only to attention value projections.
    run_temp['N_repeat'] = 2  # Recreate noisy datasets multiple times inside the same run config.
    run_temp['low_rank'] = low_rank  # LoRA rank controls the adapter capacity.
    run_temp['compute_accurate'] = True  # Try exact HVP unless the caller disables it later.
    
    runs=[]
    for run_id in range(n_runs):
        run = copy.deepcopy(run_temp) 
        run['run_id'] = run_id  # Seed and output file suffix are both derived from this index.
        runs.append(run)

    return exp, runs 

'''
config generation
'''
GLUE_TASKS = [
    "rte",
    "cola",
    "qnli",
    "qqp",
    "sst2",
    "mrpc",
    "wnli",
]

def config_qnli1():
    """Return the QNLI preset with LoRA rank 1."""
    exp, runs=generate_config(expno_name='qnli1', task='qnli', model='roberta', low_rank=1, n_runs=10)
    return exp, runs  

def config_qnli2():
    """Return the QNLI preset with LoRA rank 2."""
    exp, runs=generate_config(expno_name='qnli2', task='qnli', model='roberta', low_rank=2, n_runs=10)
    return exp, runs        

def config_qnli3():
    """Return the QNLI preset with LoRA rank 4."""
    exp, runs=generate_config(expno_name='qnli3', task='qnli', model='roberta', low_rank=4, n_runs=10)
    return exp, runs  

def config_qnli4():
    """Return the QNLI preset with LoRA rank 8."""
    exp, runs=generate_config(expno_name='qnli4', task='qnli', model='roberta', low_rank=8, n_runs=10)
    return exp, runs

def config_qnli5():
    """Return the QNLI preset with LoRA rank 16."""
    exp, runs=generate_config(expno_name='qnli5', task='qnli', model='roberta', low_rank=16, n_runs=10)
    return exp, runs

def config_qqp1():
    """Return the QQP preset with LoRA rank 1."""
    exp, runs=generate_config(expno_name='qqp1', task='qqp', model='roberta', low_rank=1, n_runs=10)
    return exp, runs  

def config_qqp2():
    """Return the QQP preset with LoRA rank 2."""
    exp, runs=generate_config(expno_name='qqp2', task='qqp', model='roberta', low_rank=2, n_runs=10)
    return exp, runs        

def config_qqp3():
    """Return the QQP preset with LoRA rank 4."""
    exp, runs=generate_config(expno_name='qqp3', task='qqp', model='roberta', low_rank=4, n_runs=10)
    return exp, runs  

def config_qqp4():
    """Return the QQP preset with LoRA rank 8."""
    exp, runs=generate_config(expno_name='qqp4', task='qqp', model='roberta', low_rank=8, n_runs=10)
    return exp, runs

def config_qqp5():
    """Return the QQP preset with LoRA rank 16."""
    exp, runs=generate_config(expno_name='qqp5', task='qqp', model='roberta', low_rank=16, n_runs=10)
    return exp, runs

def config_sst21():
    """Return the SST-2 preset with LoRA rank 1."""
    exp, runs=generate_config(expno_name='sst21', task='sst2', model='roberta', low_rank=1, n_runs=10)
    return exp, runs  

def config_sst22():
    """Return the SST-2 preset with LoRA rank 2."""
    exp, runs=generate_config(expno_name='sst22', task='sst2', model='roberta', low_rank=2, n_runs=10)
    return exp, runs        

def config_sst23():
    """Return the SST-2 preset with LoRA rank 4."""
    exp, runs=generate_config(expno_name='sst23', task='sst2', model='roberta', low_rank=4, n_runs=10)
    return exp, runs  

def config_sst24():
    """Return the SST-2 preset with LoRA rank 8."""
    exp, runs=generate_config(expno_name='sst24', task='sst2', model='roberta', low_rank=8, n_runs=10)
    return exp, runs

def config_sst25():
    """Return the SST-2 preset with LoRA rank 16."""
    exp, runs=generate_config(expno_name='sst25', task='sst2', model='roberta', low_rank=16, n_runs=10)
    return exp, runs

def config_mrpc1():
    """Return the MRPC preset with LoRA rank 1."""
    # GLUE - Microsoft Research Paraphrase Corpus
    # Determine if two sentences are paraphrases from one another or not
    exp, runs=generate_config(expno_name='mrpc1', task='mrpc', model='roberta', low_rank=1, n_runs=10)
    return exp, runs  

def config_mrpc2():
    """Return the MRPC preset with LoRA rank 2."""
    # GLUE - Microsoft Research Paraphrase Corpus
    # Determine if two sentences are paraphrases from one another or not
    exp, runs=generate_config(expno_name='mrpc2', task='mrpc', model='roberta', low_rank=2, n_runs=10)
    return exp, runs        

def config_mrpc3():
    """Return the MRPC preset with LoRA rank 4."""
    # GLUE - Microsoft Research Paraphrase Corpus
    # Determine if two sentences are paraphrases from one another or not
    exp, runs=generate_config(expno_name='mrpc3', task='mrpc', model='roberta', low_rank=4, n_runs=10)
    return exp, runs  

def config_mrpc4():
    """Return the MRPC preset with LoRA rank 8."""
    # GLUE - Microsoft Research Paraphrase Corpus
    # Determine if two sentences are paraphrases from one another or not
    exp, runs=generate_config(expno_name='mrpc4', task='mrpc', model='roberta', low_rank=8, n_runs=10)
    return exp, runs

def config_mrpc5():
    """Return the MRPC preset with LoRA rank 16."""
    # GLUE - Microsoft Research Paraphrase Corpus
    # Determine if two sentences are paraphrases from one another or not
    exp, runs=generate_config(expno_name='mrpc5', task='mrpc', model='roberta', low_rank=16, n_runs=10)
    return exp, runs

def config_wnli1():
    """Return the WNLI preset with LoRA rank 1."""
    # GLUE - Winograd Natural Language Inference
    # Determine if a sentence with an anonymous pronoun and a sentence with this pronoun replaced are entailed or not
    exp, runs=generate_config(expno_name='wnli1', task='wnli', model='roberta', low_rank=1, n_runs=10)
    return exp, runs

def config_wnli2():
    """Return the WNLI preset with LoRA rank 2."""
    # GLUE - Winograd Natural Language Inference
    # Determine if a sentence with an anonymous pronoun and a sentence with this pronoun replaced are entailed or not
    exp, runs=generate_config(expno_name='wnli2', task='wnli', model='roberta', low_rank=2, n_runs=10)
    return exp, runs

def config_wnli3():
    """Return the WNLI preset with LoRA rank 4."""
    # GLUE - Winograd Natural Language Inference
    # Determine if a sentence with an anonymous pronoun and a sentence with this pronoun replaced are entailed or not
    exp, runs=generate_config(expno_name='wnli3', task='wnli', model='roberta', low_rank=4, n_runs=10)
    return exp, runs

def config_wnli4():
    """Return the WNLI preset with LoRA rank 8."""
    # GLUE - Winograd Natural Language Inference
    # Determine if a sentence with an anonymous pronoun and a sentence with this pronoun replaced are entailed or not
    exp, runs=generate_config(expno_name='wnli4', task='wnli', model='roberta', low_rank=8, n_runs=10)
    return exp, runs

def config_wnli5():
    """Return the WNLI preset with LoRA rank 16."""
    # GLUE - Winograd Natural Language Inference
    # Determine if a sentence with an anonymous pronoun and a sentence with this pronoun replaced are entailed or not
    exp, runs=generate_config(expno_name='wnli5', task='wnli', model='roberta', low_rank=16, n_runs=10)
    return exp, runs
