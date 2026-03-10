import concurrent
import numpy as np
import sys
import os
# Go one directory up and then into the 'libs' directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import textgrad as tg
from tqdm import tqdm


def eval_sample(item, eval_fn, model):
    """Evaluate one dataset sample with the current model and return its accuracy flag."""
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")  # Wrap the raw input so TextGrad can track roles consistently.
    if isinstance(y, np.integer):
        y = int(y)
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")  # Normalize labels into TextGrad variables as well.
    response = model(x)  # Run the current prompt/model pair on a single example.

    try:
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value)
    except:
        eval_output_variable = eval_fn([x, y, response])
        eval_output_parsed = eval_fn.parse_output(eval_output_variable)

        # tmp hard code fixed unexpected response format
        # if " </ACCURACITY>" in eval_output_parsed:
        #     eval_output_parsed = eval_output_parsed.replace(" </ACCURACITY>", "")

        return int(eval_output_parsed)
    

def eval_dataset(test_set, eval_fn, model, max_samples: int=None, num_threads: int=64):
    """Evaluate a dataset, optionally truncating the sample count and parallelizing requests."""
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:  # Parallelize API-bound evaluations across samples.
        futures = []
        for _, sample in enumerate(test_set):
            
            future = executor.submit(eval_sample, sample, eval_fn, model)  # Submit each example as an independent evaluation job.
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list 

def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    """Run validation and restore the previous prompt if validation performance regresses."""
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))  # Re-score the current prompt on the validation split.
    previous_performance = np.mean(results["validation_acc"][-1])
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]
    
    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)  # Revert to the last accepted prompt when validation drops.
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)
