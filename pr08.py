import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.gridspec import GridSpec
import seaborn as sns
from datetime import datetime

# Set font and style
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def safe_torch_load(file_path):
    """Safely load .pt files, handling weights_only issue"""
    try:
        return torch.load(file_path, weights_only=True)
    except Exception as e:
        print(f"Loading with weights_only=True failed: {e}")
        print("Trying with weights_only=False...")
        try:
            return torch.load(file_path, weights_only=False)
        except Exception as e2:
            print(f"Loading with weights_only=False also failed: {e2}")
            return torch.load(file_path, map_location='cpu', weights_only=False)

def load_training_data():
    """Load training data and history"""
    model_dir = 'model_parameters'
    
    if not os.path.exists(model_dir):
        print(f"Directory {model_dir} does not exist")
        return None, None, None
    
    required_files = ['training_history.pt', 'final_quantum_weights.pt', 'initial_quantum_weights.pt']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        return None, None, None
    
    try:
        print("Loading training history...")
        training_history = safe_torch_load(f'{model_dir}/training_history.pt')
        print("Training history loaded successfully!")
        
        print("Loading final weights...")
        final_weights = safe_torch_load(f'{model_dir}/final_quantum_weights.pt')
        print("Final weights loaded successfully!")
        
        print("Loading initial weights...")
        initial_weights = safe_torch_load(f'{model_dir}/initial_quantum_weights.pt')
        print("Initial weights loaded successfully!")
        
        return training_history, final_weights, initial_weights
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

def create_output_directory():
    """Create output directory for individual charts"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'training_analysis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    return output_dir

def plot_loss_curves(training_history, output_dir):
    """Plot training and validation loss curves as separate image"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    epoch_losses = []
    val_losses = []
    
    if 'epoch_losses' in training_history:
        epoch_losses = [epoch.get('avg_loss', 0) for epoch in training_history['epoch_losses']]
    
    if 'val_losses' in training_history:
        val_losses = [val.get('val_loss', 0) for val in training_history['val_losses']]
    
    epochs = list(range(len(epoch_losses))) if epoch_losses else []
    
    if epoch_losses:
        ax.plot(epochs, epoch_losses, 'b-o', linewidth=2, markersize=6, label='Training Loss')
    if val_losses:
        ax.plot(epochs, val_losses, 'r-s', linewidth=2, markersize=6, label='Validation Loss')
    
    ax.set_xlabel('Training Epoch', fontsize=12)
    ax.set_ylabel('Loss Value', fontsize=12)
    ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    
    if epoch_losses or val_losses:
        ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add final loss annotations
    if epoch_losses:
        final_train_loss = epoch_losses[-1]
        ax.annotate(f'Final Train Loss: {final_train_loss:.6f}', 
                   xy=(epochs[-1], final_train_loss), 
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))
    
    if val_losses:
        final_val_loss = val_losses[-1]
        ax.annotate(f'Final Val Loss: {final_val_loss:.6f}', 
                   xy=(epochs[-1], final_val_loss), 
                   xytext=(10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '01_loss_curves.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path

def plot_batch_loss_heatmap(training_history, output_dir):
    """Plot batch loss heatmap as separate image"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if 'batch_losses' in training_history and training_history['batch_losses']:
        batch_losses = training_history['batch_losses']
        
        # Organize batch losses by epoch
        max_epoch = max([batch.get('epoch', 0) for batch in batch_losses])
        batch_loss_matrix = []
        
        for epoch in range(max_epoch + 1):
            epoch_batches = [batch for batch in batch_losses if batch.get('epoch', 0) == epoch]
            batch_losses_epoch = [batch.get('loss', 0) for batch in epoch_batches]
            batch_loss_matrix.append(batch_losses_epoch)
        
        # Fill to rectangular matrix
        if batch_loss_matrix:
            max_batches = max(len(row) for row in batch_loss_matrix)
            for row in batch_loss_matrix:
                while len(row) < max_batches:
                    row.append(np.nan)
            
            batch_loss_matrix = np.array(batch_loss_matrix)
            
            if batch_loss_matrix.size > 0:
                im = ax.imshow(batch_loss_matrix, cmap='viridis', aspect='auto')
                ax.set_xlabel('Batch Index')
                ax.set_ylabel('Training Epoch')
                ax.set_title('Batch Loss Heatmap', fontsize=14, fontweight='bold')
                plt.colorbar(im, ax=ax, label='Loss Value')
                
                # Add epoch labels
                ax.set_yticks(range(len(batch_loss_matrix)))
                ax.set_yticklabels([f'Epoch {i}' for i in range(len(batch_loss_matrix))])
    else:
        ax.text(0.5, 0.5, 'No batch loss data available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Batch Loss Heatmap', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '02_batch_loss_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path

def plot_weight_change_distribution(final_weights, initial_weights, output_dir):
    """Plot weight change distribution as separate image"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if final_weights is not None and initial_weights is not None:
        try:
            weight_changes = (final_weights - initial_weights).flatten().detach().numpy()
            
            # Remove outliers for better visualization
            q1, q3 = np.percentile(weight_changes, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_changes = weight_changes[(weight_changes >= lower_bound) & (weight_changes <= upper_bound)]
            
            n, bins, patches = ax.hist(filtered_changes, bins=50, alpha=0.7, 
                                      color='green', edgecolor='black')
            ax.set_xlabel('Weight Change Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Weight Change Distribution\n(Final Weights - Initial Weights)', 
                        fontsize=14, fontweight='bold')
            
            mean_change = weight_changes.mean()
            std_change = weight_changes.std()
            ax.axvline(mean_change, color='red', linestyle='--', 
                      label=f'Mean: {mean_change:.4f}\nStd: {std_change:.4f}')
            ax.legend()
            
            # Add statistics text
            stats_text = f'Total Parameters: {len(weight_changes):,}\n'
            stats_text += f'Mean Change: {mean_change:.6f}\n'
            stats_text += f'Std Change: {std_change:.6f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error in weight analysis:\n{e}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Change Distribution', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No weight data available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Weight Change Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '03_weight_change_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path

def plot_final_weight_distribution(final_weights, output_dir):
    """Plot final weight distribution as separate image"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if final_weights is not None:
        try:
            final_weights_flat = final_weights.flatten().detach().numpy()
            
            # Remove outliers for better visualization
            q1, q3 = np.percentile(final_weights_flat, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            filtered_weights = final_weights_flat[
                (final_weights_flat >= lower_bound) & (final_weights_flat <= upper_bound)
            ]
            
            n, bins, patches = ax.hist(filtered_weights, bins=50, alpha=0.7, 
                                      color='purple', edgecolor='black')
            ax.set_xlabel('Weight Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Final Weight Distribution', fontsize=14, fontweight='bold')
            
            mean_weight = final_weights_flat.mean()
            std_weight = final_weights_flat.std()
            ax.axvline(mean_weight, color='red', linestyle='--', 
                      label=f'Mean: {mean_weight:.4f}\nStd: {std_weight:.4f}')
            ax.legend()
            
            # Add statistics text
            stats_text = f'Total Parameters: {len(final_weights_flat):,}\n'
            stats_text += f'Shape: {final_weights.shape}\n'
            stats_text += f'Mean: {mean_weight:.6f}\n'
            stats_text += f'Std: {std_weight:.6f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error in weight analysis:\n{e}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Final Weight Distribution', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No final weight data available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Final Weight Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '04_final_weight_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path

def plot_weight_norm_changes(training_history, output_dir):
    """Plot weight norm changes as separate image"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if 'batch_losses' in training_history and training_history['batch_losses']:
        try:
            batch_losses = training_history['batch_losses']
            pre_norms = [batch.get('pre_weights_norm', 0) for batch in batch_losses]
            post_norms = [batch.get('post_weights_norm', 0) for batch in batch_losses]
            batch_indices = list(range(len(batch_losses)))
            
            ax.plot(batch_indices, pre_norms, 'g-', alpha=0.8, linewidth=1.5, 
                   label='Pre-update Weight Norm')
            ax.plot(batch_indices, post_norms, 'm-', alpha=0.8, linewidth=1.5, 
                   label='Post-update Weight Norm')
            
            # Add epoch separators if possible
            if batch_losses:
                epoch_changes = []
                current_epoch = batch_losses[0].get('epoch', 0)
                for i, batch in enumerate(batch_losses):
                    if batch.get('epoch', 0) != current_epoch:
                        epoch_changes.append(i)
                        current_epoch = batch.get('epoch', 0)
                
                for change_point in epoch_changes:
                    ax.axvline(change_point, color='gray', linestyle='--', alpha=0.5)
            
            ax.set_xlabel('Batch Index')
            ax.set_ylabel('Weight Norm')
            ax.set_title('Weight Norm Changes During Training', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            avg_pre_norm = np.mean(pre_norms)
            avg_post_norm = np.mean(post_norms)
            avg_change = avg_post_norm - avg_pre_norm
            
            stats_text = f'Avg Pre-norm: {avg_pre_norm:.4f}\n'
            stats_text += f'Avg Post-norm: {avg_post_norm:.4f}\n'
            stats_text += f'Avg Change: {avg_change:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error in norm analysis:\n{e}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Weight Norm Changes', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No batch data available for norm analysis', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Weight Norm Changes', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '05_weight_norm_changes.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path

def plot_loss_decrease_rate(training_history, output_dir):
    """Plot loss decrease rate as separate image"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epoch_losses = []
    if 'epoch_losses' in training_history:
        epoch_losses = [epoch.get('avg_loss', 0) for epoch in training_history['epoch_losses']]
    
    if len(epoch_losses) > 1:
        loss_decrease_rates = []
        valid_epochs = []
        
        for i in range(1, len(epoch_losses)):
            if epoch_losses[i-1] != 0:  # Avoid division by zero
                rate = (epoch_losses[i-1] - epoch_losses[i]) / epoch_losses[i-1] * 100
                loss_decrease_rates.append(rate)
                valid_epochs.append(i)
        
        if loss_decrease_rates:
            bars = ax.bar(valid_epochs, loss_decrease_rates, color='orange', alpha=0.7, 
                         edgecolor='darkorange', linewidth=1)
            ax.set_xlabel('Training Epoch')
            ax.set_ylabel('Loss Decrease Rate (%)')
            ax.set_title('Inter-epoch Loss Decrease Rate', fontsize=14, fontweight='bold')
            
            # Add value labels on bars
            for bar, rate in zip(bars, loss_decrease_rates):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{rate:.1f}%', ha='center', va='bottom' if rate >= 0 else 'top')
            
            # Add zero line and statistics
            ax.axhline(0, color='black', linestyle='-', alpha=0.3)
            
            avg_decrease = np.mean(loss_decrease_rates)
            total_decrease = sum([max(rate, 0) for rate in loss_decrease_rates])  # Only positive decreases
            
            stats_text = f'Average Decrease: {avg_decrease:.2f}%\n'
            stats_text += f'Total Decrease: {total_decrease:.2f}%\n'
            stats_text += f'Positive Epochs: {sum(1 for r in loss_decrease_rates if r > 0)}/{len(loss_decrease_rates)}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Insufficient epoch data\n(need at least 2 epochs)', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Inter-epoch Loss Decrease Rate', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '06_loss_decrease_rate.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path

def plot_data_split_info(training_history, output_dir):
    """Plot data split information as separate image"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if 'data_split_info' in training_history:
        data_info = training_history['data_split_info']
        
        labels = ['Training Set', 'Validation Set', 'Test Set']
        sizes = [
            data_info.get('train_size', 0),
            data_info.get('val_size', 0), 
            data_info.get('test_size', 0)
        ]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        
        # Only show non-zero parts
        non_zero_indices = [i for i, size in enumerate(sizes) if size > 0]
        if non_zero_indices:
            labels = [labels[i] for i in non_zero_indices]
            sizes = [sizes[i] for i in non_zero_indices]
            colors = colors[:len(sizes)]
            
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                             autopct='%1.1f%%', startangle=90,
                                             textprops={'fontsize': 12})
            ax.set_title('Data Split Ratio', fontsize=14, fontweight='bold')
            
            # Beautify pie chart
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # Add detailed information
            total_samples = sum(sizes)
            info_text = f"Total Samples: {total_samples:,}\n\n"
            for label, size in zip(labels, sizes):
                info_text += f"{label}: {size:,} ({size/total_samples*100:.1f}%)\n"
            
            # Add training samples actually used
            actual_used = data_info.get('actual_train_used', 0)
            if actual_used > 0:
                info_text += f"\nTraining Samples Used: {actual_used:,}"
            
            fig.text(0.02, 0.02, info_text, fontsize=10, 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'No data split information', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Data Split Ratio', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No data split information available', 
               ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Data Split Ratio', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '07_data_split_ratio.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path

def create_training_summary(training_history, final_weights, output_dir):
    """Create training summary as a text-based image"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Extract data
    epoch_losses = []
    val_losses = []
    if 'epoch_losses' in training_history:
        epoch_losses = [epoch.get('avg_loss', 0) for epoch in training_history['epoch_losses']]
    if 'val_losses' in training_history:
        val_losses = [val.get('val_loss', 0) for val in training_history['val_losses']]
    
    data_info = training_history.get('data_split_info', {})
    
    # Create summary text
    summary_text = "TRAINING SUMMARY REPORT\n"
    summary_text += "=" * 50 + "\n\n"
    
    summary_text += "📊 TRAINING STATISTICS\n"
    summary_text += "-" * 30 + "\n"
    summary_text += f"• Total Training Epochs: {len(epoch_losses)}\n"
    summary_text += f"• Final Training Loss: {epoch_losses[-1]:.6f}\n" if epoch_losses else "• Final Training Loss: N/A\n"
    summary_text += f"• Final Validation Loss: {val_losses[-1]:.6f}\n" if val_losses else "• Final Validation Loss: N/A\n"
    
    if len(epoch_losses) > 1:
        total_decrease = epoch_losses[0] - epoch_losses[-1]
        improvement_ratio = total_decrease / epoch_losses[0] * 100 if epoch_losses[0] != 0 else 0
        summary_text += f"• Total Loss Decrease: {total_decrease:.6f}\n"
        summary_text += f"• Relative Improvement: {improvement_ratio:.2f}%\n"
    
    summary_text += "\n🔧 MODEL INFORMATION\n"
    summary_text += "-" * 30 + "\n"
    if final_weights is not None:
        summary_text += f"• Quantum Weight Shape: {final_weights.shape}\n"
        summary_text += f"• Total Parameters: {final_weights.numel():,}\n"
        summary_text += f"• Weight Mean: {final_weights.mean():.6f}\n"
        summary_text += f"• Weight Std: {final_weights.std():.6f}\n"
    
    summary_text += "\n📁 DATA SPLIT INFORMATION\n"
    summary_text += "-" * 30 + "\n"
    summary_text += f"• Training Set: {data_info.get('train_size', 'N/A'):,} samples\n"
    summary_text += f"• Validation Set: {data_info.get('val_size', 'N/A'):,} samples\n"
    summary_text += f"• Test Set: {data_info.get('test_size', 'N/A'):,} samples\n"
    summary_text += f"• Training Samples Used: {data_info.get('actual_train_used', 'N/A'):,}\n"
    
    summary_text += "\n⏱️ TRAINING EFFICIENCY\n"
    summary_text += "-" * 30 + "\n"
    if 'batch_losses' in training_history and training_history['batch_losses']:
        batch_losses = training_history['batch_losses']
        first_epoch_batches = len([b for b in batch_losses if b.get('epoch', 0) == 0])
        summary_text += f"• Batches per Epoch: {first_epoch_batches}\n"
        if data_info.get('actual_train_used') and first_epoch_batches > 0:
            batch_size = data_info['actual_train_used'] // first_epoch_batches
            summary_text += f"• Estimated Batch Size: ~{batch_size}\n"
        summary_text += f"• Total Batches Processed: {len(batch_losses)}\n"
    
    # Add timestamp
    summary_text += f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Display text
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '08_training_summary.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path

def main():
    """Main function to generate all individual analysis charts"""
    print("Starting to load training data...")
    
    # Load data
    training_history, final_weights, initial_weights = load_training_data()
    
    if training_history is None:
        print("Unable to load training data.")
        return
    
    # Create output directory
    output_dir = create_output_directory()
    
    print("\nGenerating individual analysis charts...")
    print("=" * 60)
    
    # Generate all individual charts
    try:
        plot_loss_curves(training_history, output_dir)
        plot_batch_loss_heatmap(training_history, output_dir)
        plot_weight_change_distribution(final_weights, initial_weights, output_dir)
        plot_final_weight_distribution(final_weights, output_dir)
        plot_weight_norm_changes(training_history, output_dir)
        plot_loss_decrease_rate(training_history, output_dir)
        plot_data_split_info(training_history, output_dir)
        create_training_summary(training_history, final_weights, output_dir)
        
    except Exception as e:
        print(f"Error generating charts: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display generation summary
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE!")
    print("=" * 60)
    print(f"All charts saved to: {output_dir}")
    print("\nGenerated files:")
    generated_files = sorted(os.listdir(output_dir))
    for file in generated_files:
        file_path = os.path.join(output_dir, file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"  • {file} ({file_size:.1f} KB)")
    
    print(f"\nTotal files generated: {len(generated_files)}")
    print("Each chart is now available as a separate high-resolution PNG file.")

if __name__ == "__main__":
    main()
