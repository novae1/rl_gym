import re
import ast
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

def parse_and_plot_training_log(log_file_path):
    """
    Parse training log and create comprehensive visualization.
    
    Args:
        log_file_path: Path to the log file
        
    Returns:
        list: List of dictionaries containing the metrics
        fig: Matplotlib figure object
    """
    # Read the log file
    with open(log_file_path, 'r', encoding='utf-8') as f:
        log_text = f.read()
    
    # Extract all dictionaries from the log
    pattern = r"\{'loss':.*?\}"
    matches = re.findall(pattern, log_text, re.DOTALL)
    
    # Parse the dictionary strings
    metrics_list = []
    for match in matches:
        try:
            # Use ast.literal_eval for safe evaluation
            metrics_dict = ast.literal_eval(match)
            metrics_list.append(metrics_dict)
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Could not parse dictionary: {e}")
            continue
    
    print(f"Extracted {len(metrics_list)} metric dictionaries")
    
    # Print available keys from first dict to help debug
    if metrics_list:
        print("\nAvailable keys in first dictionary:")
        for key in sorted(metrics_list[0].keys()):
            print(f"  '{key}'")
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Extract data for plotting
    steps = list(range(len(metrics_list)))
    
    # Define subplots (4 rows x 3 columns = 12 plots)
    plots_config = [
        ('loss', 'Loss', 'tab:red'),
        ('grad_norm', 'Gradient Norm', 'tab:orange'),
        ('learning_rate', 'Learning Rate', 'tab:green'),
        ('num_tokens', 'Total Tokens', 'tab:blue'),
        ('completions/mean_length', 'Mean Completion Length', 'tab:purple'),
        ('completions/clipped_ratio', 'Clipped Ratio', 'tab:brown'),
        ('rewards/match_format_approximately/mean', 'Format Match Reward', 'tab:pink'),
        ('rewards/check_numbers/mean', 'Check Numbers Reward', 'tab:olive'),
        ('reward', 'Total Reward', 'tab:cyan'),
        ('reward_std', 'Reward Std Dev', 'tab:gray'),
        ('kl', 'KL Divergence', 'tab:red'),
        ('epoch', 'Epoch', 'tab:blue'),
    ]
    
    for idx, (key, title, color) in enumerate(plots_config, 1):
        ax = plt.subplot(4, 3, idx)
        
        # Extract values for this metric - keys with '/' are NOT nested, they're literal key names
        values = []
        for metrics in metrics_list:
            values.append(metrics.get(key, None))
        
        # Filter out None values
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        valid_steps = [steps[i] for i in valid_indices]
        valid_values = [values[i] for i in valid_indices]
        
        if valid_values:
            # Plot raw data
            ax.plot(valid_steps, valid_values, color=color, linewidth=1.5, alpha=0.4, label='Raw')
            
            # Add smoothed trend line if enough points
            if len(valid_values) > 5:
                try:
                    window_size = min(max(5, len(valid_values) // 20), len(valid_values))
                    smoothed = uniform_filter1d(valid_values, size=window_size)
                    ax.plot(valid_steps, smoothed, color=color, linewidth=2.5, alpha=1.0, label='Smoothed')
                except Exception as e:
                    print(f"Could not smooth {title}: {e}")
            
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            
            # Format y-axis for scientific notation if values are very large or very small
            if max(valid_values) > 10000 or (min(valid_values) > 0 and min(valid_values) < 0.01):
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        else:
            ax.text(0.5, 0.5, f'No data for key:\n{key}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10, color='red')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel(title, fontsize=10)
    
    plt.tight_layout()
    plt.suptitle('Training Metrics Overview', fontsize=16, fontweight='bold', y=1.002)
    
    return metrics_list, fig


# Example usage:
if __name__ == "__main__":
    # Parse and plot
    metrics_list, fig = parse_and_plot_training_log('training.log')
    
    # Save the figure
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')