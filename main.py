import os
import pickle
import tensorflow as tf
from utils.data_preprocessor import prepare_data
from training.training import run_training, plot_training_metrics
from utils.args import parse_args

def main():
    args = parse_args()

    base_dir = "results"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    for lr in args.learning_rates:
        for bs in args.batch_sizes:
            for ep in args.epochs:
                print(f"Running training with learning rate: {lr}, batch size: {bs}, epochs: {ep}")

                # Use 'bs' instead of 'BATCH_SIZE'.
                train_dataset, test_dataset, val_dataset = prepare_data(BATCH_SIZE=bs)

                # Use 'ep', 'lr', and 'bs' instead of 'EPOCHS', 'learning_rate', and 'BATCH_SIZE'.
                agent, train_loss_results, train_accuracy_results, cumulative_rewards_per_epoch, val_loss_results, val_accuracy_results, val_cumulative_rewards_per_epoch, test_loss_results, test_accuracy_results = run_training(EPOCHS=ep, learning_rate=lr, BATCH_SIZE=bs)

                # Directory structure: base_dir/lr_bs_ep/.
                save_path = os.path.join(base_dir, f"lr{lr}_bs{bs}_ep{ep}")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)

                # Save model.
                agent_save_path = os.path.join(save_path, "model")
                tf.saved_model.save(agent, agent_save_path)

                # Save metrics.
                with open(os.path.join(save_path, "train_loss_results.pkl"), "wb") as f:
                    pickle.dump(train_loss_results, f)

                with open(os.path.join(save_path, "train_accuracy_results.pkl"), "wb") as f:
                    pickle.dump(train_accuracy_results, f)

                with open(os.path.join(save_path, "cumulative_rewards_per_epoch.pkl"), "wb") as f:
                    pickle.dump(cumulative_rewards_per_epoch, f)

                # Plot and save metrics images.
                plot_training_metrics(train_loss_results, train_accuracy_results, cumulative_rewards_per_epoch,
                                      val_loss_results, val_accuracy_results, val_cumulative_rewards_per_epoch,
                                      test_loss_results, test_accuracy_results,
                                      save_path)

if __name__ == "__main__":
    main()

