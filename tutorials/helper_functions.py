import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def text_to_dict(text, attribute_class):
    """
    Convert a structured text representation to a dictionary.
    
    Args:
    text (str): The input text to convert
    attribute_class (str): The attribute class to remove from the text
    
    Returns:
    dict: A dictionary representation of the input text
    """
    # Remove attribute class and strip whitespace
    text = re.sub(attribute_class, "", text).strip()
    text = text[1:-1].strip() # remove the first and last bracket
    
    # Define the regex pattern to match key-value pairs
    pattern = r'(\w+)\[([^\]]+)\]'
    
    # Find all matches in the text
    matches = re.findall(pattern, text)
    
    # Convert matches to a dictionary, splitting values by comma and stripping whitespace
    result_dict = {key: [v.strip() for v in value.split(',')] for key, value in matches}
    
    return result_dict

def extract_function_name(text):
    """
    Extract the function name from a text string.
    
    Args:
    text (str): The input text
    
    Returns:
    str: The extracted function name
    """
    return text.split('(')[0]

def calculate_scores(responses_dict: dict):
    """
    Calculate evaluation scores for model responses.
    
    Args:
    responses_dict (dict): Dictionary containing model responses and ground truth
    
    Returns:
    tuple: A tuple containing two dictionaries (scores, avg_scores)
    """
    
    score_function_name_match = []
    score_function_attribute_match = []
    exact_match = []
    score_function_attribute_values_match = []

    for i, (k,v) in enumerate(responses_dict.items()):
        ground_truth = v['ground_truth']
        output = v['output']
        if output.startswith("Output:"): output = output.replace("Output:", "") # in llama3 models, tendency to start with "Output:"
        
        output = output.strip().lower() # remove \n from the end
        ground_truth = ground_truth.strip().lower()

        # Extract the function name
        ground_truth_fn_name = extract_function_name(ground_truth)
        output_fn_name = extract_function_name(output)
        
        # Extract the attributes and corresponding values
        ground_truth_attributes_classes = text_to_dict(ground_truth, ground_truth_fn_name)
        output_attributes_classes = text_to_dict(output, output_fn_name)
        
        # Calculate scores for different metrics
        exact_match.append(ground_truth == output)
        
        if ground_truth_fn_name == output_fn_name:
            score_function_name_match.append(True)
            
            if ground_truth_attributes_classes.keys() == output_attributes_classes.keys():
                score_function_attribute_match.append(True)
                
                score_value = all(set(ground_truth_attributes_classes[key]) == set(output_attributes_classes[key]) 
                                  for key in ground_truth_attributes_classes.keys())
                score_function_attribute_values_match.append(score_value)
            else:
                score_function_attribute_match.append(False)
                # score_function_attribute_values_match.append(False)
        else:
            score_function_name_match.append(False)
            score_function_attribute_match.append(False)
            score_function_attribute_values_match.append(False)
    
    # Compile scores and calculate averages
    scores = {
        "exact_match": exact_match,
        "function_name_match": score_function_name_match,
        "function_attribute_match": score_function_attribute_match,
        "function_attribute_values_match": score_function_attribute_values_match
    }
        
    avg_scores = {key: sum(value)/len(value) for key, value in scores.items()}
    
    return scores, avg_scores

def plot_all_model_performance(results_df):
    """
    Create a single bar plot for all metrics across all models.
    X-axis shows evaluation scores, and legends represent models.
    
    Args:
    results_df (DataFrame): DataFrame containing all model results
    
    Returns:
    matplotlib.figure.Figure: The created plot figure
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # reorder the columns
    results_df = results_df[['exact_match', 'function_name_match', 'function_attribute_match', 'function_attribute_values_match']]
    
    # Transpose the DataFrame so that metrics are columns
    results_df_t = results_df.T
    
    # Plot bars for each model
    bar_width = 0.15
    index = np.arange(len(results_df_t.index))
    for i, model in enumerate(results_df_t.columns):
        ax.bar(index + i*bar_width, results_df_t[model], bar_width, label=model)
    
    ax.set_xlabel('Evaluation Metrics', fontsize=18, labelpad=10)
    ax.set_ylabel('Score', fontsize=18)
    ax.set_title('Model Performance Across Evaluation Metrics', fontsize=20, fontweight='bold')
    
    # Set x-ticks and labels with more space
    ax.set_xticks(index + bar_width * (len(results_df_t.columns) - 1) / 2)
    ax.set_xticklabels(results_df_t.index, rotation=0, ha='center', fontsize=15)
    ax.tick_params(axis='x', which='major', pad=0)  # Add padding below x-axis labels
    
    ax.tick_params(axis='y', labelsize=16)
    
    # Add legend inside the plot
    ax.legend(title='Models', title_fontsize='16', fontsize='14', loc='upper left', bbox_to_anchor=(0.01, 0.99))
    
    plt.tight_layout()
    
    return fig
