"""End-to-end orchestration for a single classification influence run."""

from dataloader import create_dataloaders
from lora_model import LORAEngine
from influence import IFEngine
import torch

def run_experiment_core(config):
    """Train a LoRA classifier, extract gradients, and compute influence scores."""
    print(config)
    model_name_or_path=config['model_name_or_path']
    task=config['task']
    noise_ratio=config['noise_ratio']
    batch_size=config['batch_size']
    target_modules=config['target_modules']
    device=config['device']
    num_epochs=config['num_epochs']
    lr=config['lr']
    N_repeat=config['N_repeat']
    compute_accurate=config['compute_accurate']
    low_rank=config['low_rank']
    if low_rank > 4:
        compute_accurate=False  # Exact HVP becomes too expensive once the adapter rank is larger.
    
    for run_id in range(N_repeat):
        # fine-tuning models
        train_dataloader, eval_dataloader, noise_index, tokenized_datasets, collate_fn = create_dataloaders(model_name_or_path=model_name_or_path,
                                                                                                               task=task,
                                                                                                               noise_ratio=noise_ratio,
                                                                                                               batch_size=batch_size)

        lora_engine = LORAEngine(model_name_or_path=model_name_or_path,
                                    target_modules=target_modules,
                                    train_dataloader=train_dataloader,
                                    eval_dataloader=eval_dataloader,
                                    device=device,
                                    num_epochs=num_epochs,
                                    lr=lr,
                                    task=task,
                                    low_rank=low_rank)
        lora_engine.build_LORA_model()  # Attach LoRA adapters to the base classifier.
        lora_engine.train_LORA_model()  # Fit the LoRA weights on the noisy GLUE subset.
        tr_grad_dict, val_grad_dict = lora_engine.compute_gradient(tokenized_datasets, collate_fn)  # Collect per-sample gradients.

        del lora_engine, train_dataloader, eval_dataloader, tokenized_datasets, collate_fn

        # influence functions
        influence_engine = IFEngine()
        influence_engine.preprocess_gradients(tr_grad_dict, val_grad_dict, noise_index)  # Cache gradients and validation average.
        influence_engine.compute_hvps(compute_accurate=compute_accurate)  # Build inverse-Hessian-vector approximations.
        influence_engine.compute_IF()  # Convert HVPs into influence scores for all training samples.
        influence_engine.save_result(noise_index, run_id=run_id)  # Persist runtime metadata and influence arrays.

        del tr_grad_dict, val_grad_dict, noise_index, influence_engine
        with torch.no_grad():
            torch.cuda.empty_cache()  # Release GPU memory before the next repetition.


