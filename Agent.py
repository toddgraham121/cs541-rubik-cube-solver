import random
import time
from puzzle import State, move, num_solved_sides, num_pieces_correct_side, shuffle, n_move_state, one_move_state
import numpy as np
import pprint as pp


# q-values key = (state, action) => value = (q-value, update_count)
class Agent:

    # initialize agent, can be passed a dictionary of Q-Values
    # if it already exists, and a cube, otherwise, initializes new
    # cube if not provided one
    def __init__(self, QValues={}, scramble_depth: int = 3):
        # maps a state action pair to a Q-Value, and an update count for that Q-Value
        self.QV = QValues
        # create or store initial cube state, and store list of actions
        self.start_state = n_move_state(n=scramble_depth)
        print(self.start_state)
        self.curr_state = self.start_state.copy()
        self.actions = self.start_state.actions
        self.move = {"front": 0, "back": 0, "left": 0, "right": 0, "top": 0, "bottom": 0, "afront": 0, "aback": 0,
                     "aleft": 0, "aright": 0, "atop": 0, "abottom": 0}

    def adi(self, to_depth: int = 1, reward_coefficient: float = 1.0) -> None:
        goal_state = State()
        states_with_rewards = []
        states_with_rewards.append([goal_state])
        qvTable = {goal_state.__hash__(): np.zeros(12)}

        for d in range(1, to_depth + 1):
            reward = (to_depth + 1 - d)

            for s in states_with_rewards[d - 1]:
                states_at_depth = []
                for action in self.actions:
                    s_ = move(s, action)

                    if s_.__hash__() not in qvTable.keys():
                        states_at_depth.append(s_)
                        goodAction = self.reverse_action(action)
                        qvTable.update(
                            {s_.__hash__(): np.full(12, -1 * reward)})
                        qvTable[s_.__hash__()][self.actions.index(
                            goodAction)] = reward
                states_with_rewards.append(states_at_depth)

        self.QV = qvTable

    # def register_patterns(self, to_depth: int = 1, with_reward_coefficient: float = 1.0) -> None:
    #     # list of dictionaries, each dictionary a depth distance from the goal state
    #     states_with_rewards = []
    #     goal_state = State()
    #     states_with_rewards.append(
    #         {goal_state: {(goal_state.__hash__(), None): to_depth * with_reward_coefficient * 10}})
    #     for d in range(1, to_depth + 1):
    #         states_to_rewards_at_this_depth = {}
    #         reward = (to_depth + 1 - d) * with_reward_coefficient

    #         for s in states_with_rewards[d - 1]:
    #             for good_action in self.actions:
    #                 s_ = move(s, good_action)

    #                 good_action = self.reverse_action(good_action)

    #                 states_to_rewards_at_this_depth[s_] = {
    #                     (s_.__hash__(), good_action): reward}
    #                 for bad_action in self.actions:
    #                     if bad_action != good_action and (s_.__hash__(), bad_action) not in states_to_rewards_at_this_depth[s_]:
    #                         states_to_rewards_at_this_depth[s_][(
    #                             s_.__hash__(), bad_action)] = -1*reward
    #             states_with_rewards.append(states_to_rewards_at_this_depth)

    #     for state_with_reward in reversed(states_with_rewards):
    #         for state, state_action_rewards in state_with_reward.items():
    #             self.QV.update(state_action_rewards)

    def reverse_action(self, action):
        if action[0] == 'a':
            return action[1:]
        else:
            return f'a{action}'

    # explore
    def QLearn(self, gamma=0.99, steps=20, epsilon=0.9, eta=0.6):
        # execute q learning for specified number of episodes
        self.curr_state = self.start_state.copy()  # six_move_state()
        for i in range(steps):
            if not (self.curr_state.__hash__()) in self.QV.keys():
                self.QV.update({self.curr_state.__hash__(): np.zeros(12)})
            # Observe current state
            state = self.curr_state.copy()
            # Choose an action using epsilon greedy
            action = self.chooseAction(epsilon)
            # Perform the action and receive reward
            reward = self.reward(state, action)
            self.curr_state.move(self.actions[action])
            if not (self.curr_state.__hash__()) in self.QV.keys():
                self.QV.update({self.curr_state.__hash__(): np.zeros(12)})
            # Update Q Table
            self.QV[state.__hash__()][action] = self.QV[state.__hash__()][action] + eta * (
                reward + gamma*np.max(self.QV[self.curr_state.__hash__()]) - self.QV[state.__hash__()][action])

            # Check for end state
            if self.curr_state.isGoalState():
                print('Trained')
                return i
        print('Darn')
        return steps

    def chooseAction(self, epsilon=0):
        if np.random.rand() < (1 - epsilon):
            action = np.argmax(self.QV[self.curr_state.__hash__()])
        else:
            action = np.random.randint(0, 11)
        return action

    def Play(self, max_steps=20):
        self.curr_state = self.start_state.copy()

        for i in range(max_steps):
            # If the current state is not in the QV table
            if not (self.curr_state.__hash__()) in self.QV.keys():
                self.QV.update({self.curr_state.__hash__(): np.zeros(12)})

            action = self.chooseAction()
            self.curr_state.move(self.actions[action])

            if self.curr_state.isGoalState():
                print('Made it')
                return i
        return max_steps

    def reward(self, state, action):
        next_state = move(state, self.actions[action])
        if next_state.isGoalState():
            return 100
        else:
            return 0


if __name__ == '__main__':
    agent = Agent(scramble_depth=5)
    print("REGISTERING PATTERN DATABASE, THIS WILL TAKE A LITTLE WHILE")

    agent.adi(10, 1.0)

    training_episodes = 1000
    test_episodes = 2
    training_steps = np.zeros(training_episodes)
    test_steps = np.zeros(test_episodes)
    Epsilons = [i / training_episodes for i in range(training_episodes)]
    Epsilons.reverse()

    for j, e in enumerate(Epsilons):
        print("======= ROUND " + str(j) + "=========")
        training_steps[j] = agent.QLearn(steps=60, epsilon=e)
    for i in range(test_episodes):
        test_steps[i] = agent.Play(max_steps=20)

    print(training_steps)
    print(test_steps)
