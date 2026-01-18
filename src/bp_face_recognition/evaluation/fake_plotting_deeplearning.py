import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Configuration ---
epochs = 50
datasets_scenarios = {
    "webcam": {
        "initial_train_loss": 1.8, "final_train_loss": 0.1,
        "initial_val_loss": 1.9, "final_val_loss": 0.2,
        "initial_train_class_loss_factor": 0.7, "final_train_class_loss_factor": 0.6,
        "initial_val_class_loss_factor": 0.75, "final_val_class_loss_factor": 0.65,
        "initial_train_acc": 0.5, "final_train_acc": 0.97,
        "initial_val_acc": 0.45, "final_val_acc": 0.90,
        "decay_constant_loss": 5.0 / epochs,
        "growth_constant_acc": 5.0 / epochs,
        "overfit_val_loss_increase_start_epoch_factor": 0.6,
        "overfit_val_loss_increase_amount": 0.4,
        "overfit_val_acc_decrease_start_epoch_factor": 0.6,
        "overfit_val_acc_decrease_amount": 0.15,
        "noise_level_loss": 0.04,
        "noise_level_acc": 0.015,
    },
    "seccam": {
        "initial_train_loss": 2.5, "final_train_loss": 1.0,
        "initial_val_loss": 2.6, "final_val_loss": 1.1,
        "initial_train_class_loss_factor": 0.8, "final_train_class_loss_factor": 0.85,
        "initial_val_class_loss_factor": 0.8, "final_val_class_loss_factor": 0.85,
        "initial_train_acc": 0.2, "final_train_acc": 0.55,
        "initial_val_acc": 0.15, "final_val_acc": 0.50,
        "decay_constant_loss": 3.0 / epochs,
        "growth_constant_acc": 3.0 / epochs,
        "overfit_val_loss_increase_start_epoch_factor": 0.7,
        "overfit_val_loss_increase_amount": 0.1, # Less pronounced overfitting, just poor performance
        "overfit_val_acc_decrease_start_epoch_factor": 0.7,
        "overfit_val_acc_decrease_amount": 0.05,
        "noise_level_loss": 0.08,
        "noise_level_acc": 0.03,
    },
    "seccam_2": {
        "initial_train_loss": 3.0, "final_train_loss": 1.8,
        "initial_val_loss": 3.1, "final_val_loss": 2.0, # Val loss can be worse
        "initial_train_class_loss_factor": 0.9, "final_train_class_loss_factor": 0.9,
        "initial_val_class_loss_factor": 0.95, "final_val_class_loss_factor": 0.95,
        "initial_train_acc": 0.1, "final_train_acc": 0.35,
        "initial_val_acc": 0.08, "final_val_acc": 0.30,
        "decay_constant_loss": 1.5 / epochs, # Very slow learning
        "growth_constant_acc": 1.5 / epochs,
        "overfit_val_loss_increase_start_epoch_factor": 0.5, # Overfitting/divergence starts early
        "overfit_val_loss_increase_amount": 0.5,
        "overfit_val_acc_decrease_start_epoch_factor": 0.5,
        "overfit_val_acc_decrease_amount": 0.1,
        "noise_level_loss": 0.15, # More erratic
        "noise_level_acc": 0.05,  # More erratic
    }
}

# Ensure output directory exists
output_plot_dir = "dummy_deep_learning_plots"
os.makedirs(output_plot_dir, exist_ok=True)
output_csv_dir = "dummy_deep_learning_csvs"
os.makedirs(output_csv_dir, exist_ok=True)


for dataset_name, params in datasets_scenarios.items():
    print(f"Generating data for: {dataset_name}")
    history_data = {
        'loss': [],
        'val_loss': [],
        'classification_output_loss': [],
        'val_classification_output_loss': [],
        'classification_output_accuracy': [],
        'val_classification_output_accuracy': []
    }

    # --- TOTAL LOSS ---
    base_train_loss = (params["initial_train_loss"] - params["final_train_loss"]) * \
                      np.exp(-params["decay_constant_loss"] * np.arange(epochs)) + \
                      params["final_train_loss"] + \
                      np.random.normal(0, params["noise_level_loss"], epochs)
    
    base_val_loss = (params["initial_val_loss"] - params["final_val_loss"]) * \
                    np.exp(-params["decay_constant_loss"] * np.arange(epochs) * 0.95) + \
                    params["final_val_loss"] + \
                    np.random.normal(0, params["noise_level_loss"] * 1.2, epochs) # Slightly more noise for val

    base_val_loss = np.maximum(base_val_loss, base_train_loss * 0.85) # Val loss generally not much better than train
    overfit_start_epoch_loss = int(epochs * params["overfit_val_loss_increase_start_epoch_factor"])
    base_val_loss[overfit_start_epoch_loss:] += np.linspace(0, params["overfit_val_loss_increase_amount"], epochs - overfit_start_epoch_loss)

    history_data['loss'] = np.clip(base_train_loss, params["final_train_loss"] * 0.8, params["initial_train_loss"] * 1.1).tolist()
    history_data['val_loss'] = np.clip(base_val_loss, params["final_val_loss"] * 0.8, params["initial_val_loss"] * 1.2).tolist()

    # --- CLASSIFICATION LOSS (derived but with own characteristics) ---
    initial_train_class_loss = params["initial_train_loss"] * params["initial_train_class_loss_factor"]
    final_train_class_loss = params["final_train_loss"] * params["final_train_class_loss_factor"]
    initial_val_class_loss = params["initial_val_loss"] * params["initial_val_class_loss_factor"]
    final_val_class_loss = params["final_val_loss"] * params["final_val_class_loss_factor"]

    base_train_class_loss = (initial_train_class_loss - final_train_class_loss) * \
                            np.exp(-params["decay_constant_loss"]*1.05 * np.arange(epochs)) + \
                            final_train_class_loss + \
                            np.random.normal(0, params["noise_level_loss"]*0.9, epochs)
    
    base_val_class_loss = (initial_val_class_loss - final_val_class_loss) * \
                          np.exp(-params["decay_constant_loss"]*1.0 * np.arange(epochs)) + \
                          final_val_class_loss + \
                          np.random.normal(0, params["noise_level_loss"]*1.1, epochs)

    base_val_class_loss = np.maximum(base_val_class_loss, base_train_class_loss * 0.8)
    base_val_class_loss[overfit_start_epoch_loss:] += np.linspace(0, params["overfit_val_loss_increase_amount"]*0.8, epochs - overfit_start_epoch_loss)


    history_data['classification_output_loss'] = np.clip(base_train_class_loss, final_train_class_loss * 0.7, initial_train_class_loss * 1.1).tolist()
    history_data['val_classification_output_loss'] = np.clip(base_val_class_loss, final_val_class_loss * 0.7, initial_val_class_loss * 1.2).tolist()
    
    # --- ACCURACY ---
    base_train_acc = params["final_train_acc"] - \
                     (params["final_train_acc"] - params["initial_train_acc"]) * \
                     np.exp(-params["growth_constant_acc"] * np.arange(epochs)) + \
                     np.random.normal(0, params["noise_level_acc"], epochs)
    
    base_val_acc = params["final_val_acc"] - \
                   (params["final_val_acc"] - params["initial_val_acc"]) * \
                   np.exp(-params["growth_constant_acc"] * np.arange(epochs) * 0.95) + \
                   np.random.normal(0, params["noise_level_acc"] * 1.2, epochs)

    base_train_acc = np.clip(base_train_acc, 0.01, 0.99) # Clip accuracy
    base_val_acc = np.clip(base_val_acc, 0.01, 0.99)

    base_val_acc = np.minimum(base_val_acc, base_train_acc * 1.05) # Val acc generally not much better
    overfit_start_epoch_acc = int(epochs * params["overfit_val_acc_decrease_start_epoch_factor"])
    base_val_acc[overfit_start_epoch_acc:] -= np.linspace(0, params["overfit_val_acc_decrease_amount"], epochs - overfit_start_epoch_acc)
    base_val_acc = np.clip(base_val_acc, 0.01, 0.99) # Re-clip after potential decrease

    history_data['classification_output_accuracy'] = base_train_acc.tolist()
    history_data['val_classification_output_accuracy'] = base_val_acc.tolist()

    # --- Save to CSV ---
    deep_learning_history_df = pd.DataFrame(history_data)
    csv_filename = f"dummy_deep_learning_history_{dataset_name}.csv"
    csv_path_deep_learning = os.path.join(output_csv_dir, csv_filename)
    deep_learning_history_df.to_csv(csv_path_deep_learning, index_label="epoch")
    print(f"Dummy history data for {dataset_name} saved to {csv_path_deep_learning}")

    # --- Plotting ---
    fig, ax = plt.subplots(ncols=3, figsize=(20,5))
    fig.suptitle(f'Training History: {dataset_name}', fontsize=16)

    ax[0].plot(history_data['loss'], color='teal', label='loss')
    ax[0].plot(history_data['val_loss'], color='orange', label='val loss')
    ax[0].set_title('Total Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.7)

    ax[1].plot(history_data['classification_output_loss'], color='teal', label='class loss')
    ax[1].plot(history_data['val_classification_output_loss'], color='orange', label='val class loss')
    ax[1].set_title('Classification Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid(True, linestyle='--', alpha=0.7)

    ax[2].plot(history_data['classification_output_accuracy'], color='teal', label='accuracy')
    ax[2].plot(history_data['val_classification_output_accuracy'], color='orange', label='val accuracy')
    ax[2].set_title('Classification Accuracy')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Accuracy')
    ax[2].legend()
    ax[2].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plot_filename = f"dummy_deep_learning_plots_{dataset_name}.png"
    plot_path_deep_learning = os.path.join(output_plot_dir, plot_filename)
    plt.savefig(plot_path_deep_learning)
    print(f"Dummy plots for {dataset_name} saved to {plot_path_deep_learning}")
    plt.show()
    plt.close(fig) # Close the figure to free memory

print("\nAll dummy data and plots generated.")