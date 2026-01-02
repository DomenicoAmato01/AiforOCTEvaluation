import json
import os

def aggregate_results(base_path="results/cyst-model", output_filename="cyst-results.json"):
    """
    Aggregates evaluation results from individual model JSON files into a single JSON file.

    Args:
        base_path (str): The base directory where individual model result folders are located.
        output_filename (str): The name of the aggregated output JSON file.
    """
    all_aggregated_results = {}
    
    # Iterate through each model directory
    for model_name in os.listdir(base_path):
        model_dir = os.path.join(base_path, model_name)
        if os.path.isdir(model_dir):
            eval_results_path = os.path.join(model_dir, "eval_results.json")
            if os.path.exists(eval_results_path):
                with open(eval_results_path, "r") as f:
                    model_results = json.load(f)
                    # Assuming eval_results.json contains a dictionary where keys are model names
                    # and values are their respective evaluation metrics (e.g., accuracy)
                    all_aggregated_results.update(model_results)
            else:
                print(f"Warning: eval_results.json not found in {model_dir}")

    output_path = os.path.join(base_path, output_filename)
    with open(output_path, "w") as f:
        json.dump(all_aggregated_results, f, indent=4)
    print(f"Aggregated results saved to {output_path}")

if __name__ == "__main__":
    aggregate_results()
