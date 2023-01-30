import gym
import random

# Initialize the environment
env = gym.make('CartPole-v1')

# Define the learning algorithm
def reinforcement_learning(env, num_episodes):
    # Define the parameters
    best_weights = None
    best_reward = 0
    for i in range(num_episodes):
        # Randomly initialize the weights
        weights = [random.uniform(-1, 1) for _ in range(4)]

        # Initialize the episode
        observation = env.reset()
        total_reward = 0
        done = False

        # Run the episode
        while not done:
            # Calculate the Q-value for each action
            q_values = [sum([weights[j] * observation[j] for j in range(4)]) for _ in range(2)]
            action = int(q_values[0] > q_values[1])

            # Take the action
            observation, reward, done, info = env.step(action)
            total_reward += reward

        # Update the best weights
        if total_reward > best_reward:
            best_weights = weights
            best_reward = total_reward

    # Return the best weights
    return best_weights

# Train the model
weights = reinforcement_learning(env, 1000)

# Test the model
observation = env.reset()
total_reward = 0
done = False
while not done:
    q_values = [sum([weights[j] * observation[j] for j in range(4)]) for _ in range(2)]
    action = int(q_values[0] > q_values[1])
    observation, reward, done, info = env.step(action)
    total_reward += reward

print("Total reward:", total_reward)
