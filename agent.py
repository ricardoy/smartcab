import random
import argparse
import sys
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np


class QValue(object):
    """The Q-Value"""

    def __init__(self, state, actions, epsilon):
        self.total_occurences = 0
        self.state = state
        self.scores = dict()
        self.epsilon = epsilon
        self.actions = actions
        for a in actions:
            self.scores[a] = 0.

    def get_epsilon_and_update(self):
        result = self.epsilon
        self.epsilon = self.epsilon * 0.9
        return result

    def update(self, action, score):
        self.total_occurences = self.total_occurences + 1
        self.scores[action] = score

    def score(self, action):
        return self.scores[action]

    def max(self):
        max_score = None
        max_action = None
        for a in self.actions:
            if (max_score is None or self.scores[a] > max_score):
                max_score = self.scores[a]
                max_action = a
        return (max_score, max_action)

    def get_best_action(self):
        (score, action) = self.max()
        return action

    def get_max_value(self):
        (score, action) = self.max()
        return score

    def __str__(self):
        return '{} {}'.format(self.state, self.scores)


class TrialResultKeeper(object):
    """Will keep record of all average-rewards for the agent """

    def __init__(self):
        self.average_rewards = []
        self.goal_reached = []
        self.last_reward = None
        self.current_sum = 0
        self.current_n = 0

    def reset(self):
        if (self.current_n > 0):
            self.average_rewards.append(1.0 * self.current_sum / self.current_n)
            if (self.last_reward > 9):
                self.goal_reached.append(True)
            else:
                self.goal_reached.append(False)
        self.current_sum = 0
        self.current_n = 0
        self.last_reward = None

    def add_reward(self, reward):
        self.last_reward = reward
        self.current_sum = self.current_sum + reward
        self.current_n = self.current_n + 1


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env, alpha=0.9, gamma=0.9, epsilon=1):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actions = [None, 'left', 'forward', 'right']
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.inputs = None

        self.q = dict()
        self.last_reward = None
        self.last_state = None
        self.last_action = None
        self.trial_result_keeper = TrialResultKeeper()

    def select_action_greedy_policy(self, q):
        if (random.uniform(0, 1) < q.get_epsilon_and_update()):
            return (random.choice(self.actions), 'random')
        else:
            (score, action) = q.max()
            return (action, 'max_qvalue')

    def get_q(self, state):
        if (state in self.q):
            return self.q[state]
        else:
            return self.q.setdefault(state, QValue(state, self.actions, self.epsilon))

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.trial_result_keeper.reset()
        self.last_reward = None
        self.last_state = None
        self.last_action = None
        print 'Reset {}'.format(destination)

    def pretty_print(self, q, alpha, r, gamma, max_q):
        print '{} + {}({} + {}*{} - {})'.format(q, alpha, r, gamma, max_q, q)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        self.inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        state = self.generate_state()
        q = self.get_q(state)

        # TODO: Select action according to your policy
        (action, selection_rule) = self.select_action_greedy_policy(q)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.trial_result_keeper.add_reward(reward)

        # TODO: Learn policy based on state, action, reward
        if (self.last_reward is not None):
            last_q = self.get_q(self.last_state)
            # self.pretty_print(last_q.score(self.last_action), self.alpha, self.last_reward, self.gamma, q.get_max_value())

            score = last_q.score(self.last_action) + \
                self.alpha * (self.last_reward + self.gamma * q.get_max_value() - last_q.score(self.last_action))
            last_q.update(self.last_action, score)
            self.epsilon = self.epsilon

        print "LearningAgent.update(): deadline = {}, inputs = {}, next_waypoint = {}, action = {}, selection={}, reward = {}" \
            .format(deadline, self.inputs, self.next_waypoint, action, selection_rule, reward)  # [debug]

        self.last_action = action
        self.last_state = state
        self.last_reward = reward

    def generate_state(self):
        return (self.inputs['light'], self.inputs['oncoming'], self.inputs['right'], self.inputs['left'], self.next_waypoint)

    def final_ten_trials_scores(self):
        rewards = self.trial_result_keeper.average_rewards
        goal_reached = self.trial_result_keeper.goal_reached

        avg = 0.
        success = 0.
        for i in range(-1, -11, -1):
            avg = avg + rewards[i]
            if (goal_reached[i]):
                success = success + 1

        return (avg / 10., success / 10.)


def output_evaluate(average_rewards):
    limit = int(len(average_rewards) * 0.9)
    first_runs = []
    for i in range(0, limit):
        first_runs.append('%.1f' % (average_rewards[i]))
    print 'Average Reward for first 90% runs:'
    print first_runs

    last_runs = []
    for i in range(limit, len(average_rewards)):
        last_runs.append('%.1f' % (average_rewards[i]))
    print 'Average Reward for last 10% runs:'
    print last_runs


def pretty_print_goal_reached(x):
    if (x):
        return '+'
    else:
        return '.'


def compare_best(agent):
    total_actions = 0
    equal_actions = 0
    total_weighted_actions = 0
    equal_weighted_actions = 0
    for light in ['red', 'green']:
        for left in [None, 'left', 'forward', 'right']:
            for right in [None, 'left', 'forward', 'right']:
                for oncoming in [None, 'left', 'forward', 'right']:
                    for next_waypoint in ['left', 'forward', 'right']:
                        perfect_choice = perfect_agent_choice(light, oncoming, right, left, next_waypoint)

                        q = agent.get_q((light, oncoming, right, left, next_waypoint))
                        agent_choice = q.get_best_action()

                        if (q.total_occurences > 0):
                            print 'estado: (light={} left={} oncoming={} right={} next={}) escolha_perfeita={} escolha_agente={} total={}'.format(light, left, oncoming, right, next_waypoint, perfect_choice, agent_choice, q.total_occurences)
                            if (perfect_choice == agent_choice):
                                equal_actions = equal_actions + 1
                                equal_weighted_actions = equal_weighted_actions + q.total_occurences
                            total_actions = total_actions + 1
                            total_weighted_actions = total_weighted_actions + q.total_occurences
    return (equal_actions / float(total_actions), equal_weighted_actions / float(total_weighted_actions))


def perfect_agent_choice(light, oncoming, right, left, next_waypoint):
    if (light == 'green'):
        if (next_waypoint == 'left'):
            if (oncoming is None or oncoming == 'left'):
                return next_waypoint
            else:
                return None
        else:
            return next_waypoint
    else:
        if (next_waypoint == 'right'):
            if (left != 'forward'):
                return next_waypoint
            else:
                return None
        else:
            return None


def output_goal_reached(goal_reached):
    print 'Goal reached (. no, + yes): {}'.format(''.join(pretty_print_goal_reached(x) for x in goal_reached))


def run(num_dummies, alpha, gamma, epsilon):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(num_dummies)  # create environment (also adds some dummy traffic)
    agent = e.create_agent(LearningAgent, alpha, gamma, epsilon)  # create agent
    e.set_primary_agent(agent, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    # This reset is used to save the result of the last trial in AverageRewardKeeper
    agent.reset()

    output_evaluate(agent.trial_result_keeper.average_rewards)
    output_goal_reached(agent.trial_result_keeper.goal_reached)
    return agent


def run_no_display(num_dummies, alpha, gamma, epsilon):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment(num_dummies)  # create environment (also adds some dummy traffic)
    agent = e.create_agent(LearningAgent, alpha, gamma, epsilon)  # create agent
    e.set_primary_agent(agent, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    # This reset is used to save the result of the last trial in AverageRewardKeeper
    agent.reset()

    output_evaluate(agent.trial_result_keeper.average_rewards)
    output_goal_reached(agent.trial_result_keeper.goal_reached)
    return agent


def error_print(str):
    print >> sys.stderr, str


def grid_search():
    limit = 10
    for alpha in np.arange(0.1, 1.1, 0.1):
        for gamma in np.arange(0.0, 1.1, 0.1):
            for epsilon in np.arange(0.0, 1.1, 0.1):
                avg_score = .0
                avg_successful_runs = .0
                for _ in range(0, limit):
                    agent = run_no_display(3, alpha, gamma, epsilon)
                    (score, successful_runs) = agent.final_ten_trials_scores()
                    avg_score = avg_score + score
                    avg_successful_runs = avg_successful_runs + successful_runs
                avg_score = avg_score / float(limit)
                avg_successful_runs = avg_successful_runs / float(limit)
                error_print('%.5f %.2f %.1f %.1f %.1f' % (avg_score, avg_successful_runs, alpha, gamma, epsilon))


if __name__ == '__main__':
    # random.seed(0)
    parser = argparse.ArgumentParser(description='Smart Cab Project.')
    parser.add_argument('-alpha', type=float, default=0.1)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-epsilon', type=float, default=1.0)
    parser.add_argument('-num-dummies', type=int, default=3)
    parser.add_argument('--grid-search', action='store_true')
    parser.add_argument('--compare-best', action='store_true')

    args = parser.parse_args()

    if (args.grid_search):
        grid_search()
    else:
        if (args.compare_best):
            agent = run_no_display(args.num_dummies, args.alpha, args.gamma, args.epsilon)
            print '{}'.format(compare_best(agent))
        else:
            agent = run(args.num_dummies, args.alpha, args.gamma, args.epsilon)
