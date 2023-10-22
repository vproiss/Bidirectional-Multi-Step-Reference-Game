
import os
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from utils.args import parse_args
from utils.data_preprocessor import prepare_data
from configs.settings import SCALING_FACTOR, COM_FEATURE_SIZE
from models.communication import CommunicationAgent

sns.set(style="whitegrid")


args = parse_args()

def run_training(learning_rate=args.learning_rates[0], EPOCHS=args.epochs[0], BATCH_SIZE=args.batch_sizes[0]):
    train_dataset, val_dataset, test_dataset = prepare_data(BATCH_SIZE=BATCH_SIZE)

    def compute_reward(target, prediction_output):
        # Convert tensors to float32.
        target = tf.cast(target, dtype=tf.float32)
        prediction_output = tf.cast(prediction_output, dtype=tf.float32)
        # Calculate binary cross entropy loss.
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = bce(target, prediction_output)
        # Scale the loss by a factor and return negative loss as reward.
        scaled_loss = loss * SCALING_FACTOR
        return -scaled_loss

    def train_step(agent: CommunicationAgent, 
                   optimizer: tf.keras.optimizers.Optimizer, 
                   R1: tf.Tensor, t1: tf.Tensor, 
                   R2: tf.Tensor, t2: tf.Tensor, 
                   init_states: Tuple[tf.Tensor, tf.Tensor] = None, 
                   NUM_STEPS=5, train=True) -> Tuple[tf.Tensor, Tuple[tf.Tensor, tf.Tensor], List[tf.Tensor]]:
        # Initialize lists to store log probabilities, rewards, and predictions.
        total_log_probs, total_rewards = [], []
        all_predictions = []
        
        with tf.GradientTape() as tape:
            message_gen_states, agent_core_states = init_states

            for step in range(NUM_STEPS):
                # Switch roles of sender and receiver based on step.
                if step % 2 == 0:
                    sender, receiver = R1, R2
                    target = t1
                else:
                    sender, receiver = R2, R1
                    target = t2

                # Call the agent to get message and prediction distributions and updated states.
                message_distribution, prediction_distribution, (message_gen_states, agent_core_states) = agent(sender, receiver, states=(message_gen_states, agent_core_states))
                
                # Sample message and calculate log probability.
                message_distrib = tfp.distributions.Categorical(probs=message_distribution)
                message = message_distrib.sample()
                message_log_prob = tf.reduce_sum(message_distrib.log_prob(message), axis=1)
                
                # Sample prediction and calculate log probability.
                prediction_distrib = tfp.distributions.Bernoulli(probs=prediction_distribution)
                prediction = prediction_distrib.sample()
                prediction_log_prob = tf.reduce_sum(prediction_distrib.log_prob(prediction), axis=1)
                
                # Calculate total log probability and reward.
                total_action_log_probabilities = message_log_prob + prediction_log_prob
                reward = compute_reward(target, prediction)
                
                # Append results to lists.
                total_log_probs.append(total_action_log_probabilities)
                total_rewards.append(reward)
                all_predictions.append(prediction)

            # Calculate total loss as negative mean of product of log probabilities and rewards.
            loss = -tf.reduce_mean([tf.reduce_sum(lp * r) for lp, r in zip(total_log_probs, total_rewards)])

        # If training, calculate gradients and update weights.
        if train:
            gradients = tape.gradient(loss, agent.trainable_variables)
            optimizer.apply_gradients(zip(gradients, agent.trainable_variables))
        
        # Return loss, updated states, and predictions.
        updated_states = (message_gen_states, agent_core_states)

        return loss, updated_states, all_predictions
    
    # Function to evaluate agent on a dataset.
    def evaluate_agent(agent, dataset, BATCH_SIZE):
        # Initialize new states for validation or test.
        val_message_states = tf.zeros((BATCH_SIZE, COM_FEATURE_SIZE))
        val_agent_states = tf.zeros((BATCH_SIZE, COM_FEATURE_SIZE))

        # Initialize variables to store total loss, cumulative reward, and counts for metrics.
        total_loss, cumulative_reward = 0.0, 0.0
        total_predictions, correct_predictions, batch_count = 0, 0, 0
        
        # Loop through each batch in the dataset.
        for i, (R1, t1, R2, t2) in enumerate(dataset):
            # Call train_step with train=False to get loss, updated states, and predictions without updating weights.
            loss, _, all_predictions_output = train_step(agent, optimizer, R1, t1, R2, t2, init_states=(val_message_states, val_agent_states), train=False)

            # Update total loss.
            total_loss += loss.numpy()
            batch_count += 1 

            # Loop through each prediction output.
            for j, prediction_output in enumerate(all_predictions_output):
                if j % 2 == 0:
                    target = t1
                else:
                    target = t2

                # Calculate the number of correct predictions.
                correct_prediction = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.round(target), tf.int32), tf.cast(tf.round(prediction_output), tf.int32)), tf.float32))
                correct_predictions += correct_prediction.numpy()
                total_predictions += target.shape[0]

                # Calculate the reward for the predictions.
                reward = compute_reward(target, prediction_output)
                cumulative_reward += tf.reduce_sum(reward).numpy()

        # Calculate average loss, accuracy, and cumulative reward over all batches.
        average_loss = total_loss / batch_count
        average_accuracy = correct_predictions / total_predictions
        average_cumulative_reward = cumulative_reward / batch_count

        return average_loss, average_accuracy, average_cumulative_reward
    
    # Initialize the communication agent and the optimizer.
    agent = CommunicationAgent()
    # Use 'tf.keras.optimizers.legacy.Adam' if training on M1/M2 mac.
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    
    # Initialize lists to store training, validation, and test metrics.
    train_loss_results = []
    train_accuracy_results = []
    cumulative_rewards_per_epoch = []

    val_loss_results = []
    val_accuracy_results = []
    val_cumulative_rewards_per_epoch = []

    test_loss_results = []
    test_accuracy_results = []
    test_cumulative_rewards_per_epoch = []

    # Initialize states for the message and agent.
    message_states = None
    agent_states = None

    for epoch in range(EPOCHS):
        # Initialize variables for cumulative metrics.
        total_loss, cumulative_reward = 0.0, 0.0
        total_predictions, correct_predictions, batch_count = 0, 0, 0

        # Initialize progress bar.
        pbar = tqdm(train_dataset, desc=f"Epoch {epoch}")

        # Loop through each batch in the training dataset.
        for i, (R1, t1, R2, t2) in enumerate(pbar):
            # Perform a training step and retrieve loss, updated states, and predictions.
            loss, updated_states, all_predictions_output = train_step(
                agent, optimizer, R1, t1, R2, t2, init_states=(message_states, agent_states))

            # Update cumulative metrics.
            total_loss += loss.numpy()
            message_states, agent_states = updated_states
            batch_count += 1 

            # Loop through each prediction output and calculate accuracy and reward.
            for j, prediction_output in enumerate(all_predictions_output):
                if j % 2 == 0:
                    target = t1
                else:
                    target = t2

                correct_prediction = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.round(target), tf.int32), tf.cast(tf.round(prediction_output), tf.int32)), tf.float32))
                correct_predictions += correct_prediction.numpy()
                total_predictions += target.shape[0]

                reward = compute_reward(target, prediction_output)
                cumulative_reward += tf.reduce_sum(reward).numpy()
            
            # Update progress bar with current metrics.
            pbar.set_postfix({
                "Average Loss": total_loss / batch_count, 
                "Accuracy": correct_predictions / total_predictions,
                "Cumulative Reward": cumulative_reward / batch_count
            })

        # Calculate average metrics for the epoch.
        average_loss = total_loss / batch_count
        average_accuracy = correct_predictions / total_predictions
        average_cumulative_reward = cumulative_reward / batch_count
        
        # Append average metrics to training lists.
        train_loss_results.append(average_loss)
        train_accuracy_results.append(average_accuracy)
        cumulative_rewards_per_epoch.append(average_cumulative_reward)

        # Log training metrics.
        print(f'Epoch {epoch}, Training Loss: {average_loss:.2f}, Training Accuracy: {average_accuracy:.2f}%, Training Cumulative Reward: {average_cumulative_reward:.2f}')

        # Evaluate the agent on the validation dataset and append metrics to validation lists.
        val_loss, val_accuracy, val_cumulative_reward = evaluate_agent(agent, val_dataset, BATCH_SIZE)
        print(f'Epoch {epoch}, Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_accuracy:.2f}%, Validation Cumulative Reward: {val_cumulative_reward:.2f}')
        val_loss_results.append(val_loss)
        val_accuracy_results.append(val_accuracy)
        val_cumulative_rewards_per_epoch.append(val_cumulative_reward)

        # Evaluate the agent on the test dataset and append metrics to test lists.
        test_loss, test_accuracy, test_cumulative_reward = evaluate_agent(agent, test_dataset, BATCH_SIZE)
        print(f'Epoch {epoch}, Test Loss: {test_loss:.2f}, Test Accuracy: {test_accuracy:.2f}%, Test Cumulative Reward: {test_cumulative_reward:.2f}')
        test_loss_results.append(test_loss)
        test_accuracy_results.append(test_accuracy)
        test_cumulative_rewards_per_epoch.append(test_cumulative_reward)

    return agent, train_loss_results, train_accuracy_results, cumulative_rewards_per_epoch, val_loss_results, val_accuracy_results, val_cumulative_rewards_per_epoch, test_loss_results, test_accuracy_results, test_cumulative_rewards_per_epoch

# Function to plot the training metrics.
def plot_training_metrics(train_loss_results, train_accuracy_results, cumulative_rewards_per_epoch,
                          val_loss_results, val_accuracy_results, val_cumulative_rewards_per_epoch,
                          test_loss_results, test_accuracy_results, test_cumulative_rewards_per_epoch,
                          save_path):
    epochs = range(len(train_loss_results))

    fig, ax1 = plt.subplots(figsize=(12, 6))

    sns.lineplot(x=epochs, y=train_loss_results, label='Train Loss', ax=ax1)
    sns.lineplot(x=epochs, y=val_loss_results, label='Validation Loss', ax=ax1)
    sns.lineplot(x=epochs, y=test_loss_results, label='Test Loss', ax=ax1)

    ax2 = ax1.twinx()
    sns.lineplot(x=epochs, y=train_accuracy_results, label='Train Accuracy', ax=ax2, linestyle='dashed')
    sns.lineplot(x=epochs, y=val_accuracy_results, label='Validation Accuracy', ax=ax2, linestyle='dashed')
    sns.lineplot(x=epochs, y=test_accuracy_results, label='Test Accuracy', ax=ax2, linestyle='dashed')

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'loss_and_accuracy_plot.png'))
    plt.show()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    sns.lineplot(x=epochs, y=cumulative_rewards_per_epoch, label='Train Cumulative Reward', ax=ax1)
    sns.lineplot(x=epochs, y=val_cumulative_rewards_per_epoch, label='Validation Cumulative Reward', ax=ax1)
    sns.lineplot(x=epochs, y=test_cumulative_rewards_per_epoch, label='Test Cumulative Reward', ax=ax1)

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Cumulative Reward')

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'cumulative_reward_plot.png'))
    plt.show()