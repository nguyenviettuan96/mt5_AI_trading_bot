import itertools
import logging
import numpy as np
import tensorflow
import tensorflow.compat.v1 as tf
import time
import os
import pandas as pd
import subprocess


def check_dir(cur_dir):
    if not os.path.exists(cur_dir):
        return False
    return True


def copy_file(src_dir, tar_dir):
    cmd = 'cp %s %s' % (src_dir, tar_dir)
    subprocess.check_call(cmd, shell=True)


def find_file(cur_dir, suffix='.ini'):
    for file in os.listdir(cur_dir):
        if file.endswith(suffix):
            return cur_dir + '/' + file
    logging.error('Cannot find %s file' % suffix)
    return None


def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


def init_log(log_dir):
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                        level=logging.INFO,
                        handlers=[
                            logging.FileHandler('%s/%d.log' %
                                                (log_dir, time.time())),
                            logging.StreamHandler()
                        ])


def init_test_flag(test_mode):
    if test_mode == 'no_test':
        return False, False
    if test_mode == 'in_train_test':
        return True, False
    if test_mode == 'after_train_test':
        return False, True
    if test_mode == 'all_test':
        return True, True
    return False, False


class Counter:
    def __init__(self, total_step, test_step, log_step):
        self.counter = itertools.count(1)
        self.cur_step = 0
        self.cur_test_step = 0
        self.total_step = total_step
        self.test_step = test_step
        self.log_step = log_step
        self.stop = False
        # self.init_test = True

    def next(self):
        self.cur_step = next(self.counter)
        return self.cur_step

    def should_test(self):
        test = False
        if (self.cur_step - self.cur_test_step) >= self.test_step:
            test = True
            self.cur_test_step = self.cur_step
        return test

    def should_log(self):
        return (self.cur_step % self.log_step == 0)

    def should_stop(self):
        if self.cur_step >= self.total_step:
            return True
        return self.stop


class Trainer():
    def __init__(self, env, model, global_counter, summary_writer, run_test, output_path=None):
        self.cur_step = 0
        self.global_counter = global_counter
        self.env = env
        self.agent = self.env.agent
        self.model = model
        self.sess = self.model.sess
        self.n_step = self.model.n_step
        self.summary_writer = summary_writer
        self.run_test = run_test
        self.data = []
        self.output_path = output_path
        if run_test:
            self.test_num = self.env.test_num
            logging.info('Testing: total test num: %d' % self.test_num)
        self._init_summary()

    def _init_summary(self):
        self.train_reward = tf.placeholder(tensorflow.float32, [])
        self.train_summary = tf.summary.scalar(
            'train_reward', self.train_reward)
        self.test_reward = tf.placeholder(tensorflow.float32, [])
        self.test_summary = tf.summary.scalar('test_reward', self.test_reward)

    def _add_summary(self, reward, global_step, is_train=True):
        if is_train:
            summ = self.sess.run(self.train_summary, {
                                 self.train_reward: reward})
        else:
            summ = self.sess.run(self.test_summary, {self.test_reward: reward})
        self.summary_writer.add_summary(summ, global_step=global_step)

    def explore(self, prev_ob, prev_done):
        ob = prev_ob
        done = prev_done
        rewards = []
        for _ in range(self.n_step):
            policy, value = self.model.forward(ob, done)
            # need to update fingerprint before calling step
            self.env.update_fingerprint(policy)
            action = []
            for pi in policy:
                action.append(np.random.choice(
                    np.arange(len(pi)), p=pi))

            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            global_step = self.global_counter.next()
            self.cur_step += 1
            if self.agent.endswith('a2c'):
                self.model.add_transition(ob, action, reward, value, done)
            else:
                self.model.add_transition(ob, action, reward, next_ob, done)
            # logging
            if self.global_counter.should_log():
                logging.info('''Training: global step %d, episode step %d,
                                   ob: %s, a: %s, pi: %s, r: %.2f, train r: %.2f, done: %r''' %
                             (global_step, self.cur_step,
                              str(ob), str(action), str(policy), global_reward, np.mean(reward), done))

            if done:
                break
            ob = next_ob
        if self.agent.endswith('a2c'):
            if done:
                R = 0 if self.agent == 'a2c' else [0] * self.model.n_agent
            else:
                R = self.model.forward(ob, False, 'v')
        else:
            R = 0
        return ob, done, R, rewards

    def inference(self, ob, policy_type='default'):
        # note this done is pre-decision to reset LSTM states!
        done = False
        # self.model.reset()
        # policy-based on-poicy learning
        policy = self.model.forward(ob, done, 'p')
        self.env.update_fingerprint(policy)
        action = []
        for pi in policy:
            if policy_type != 'deterministic':
                action.append(np.random.choice(
                    np.arange(len(pi)), p=pi))
            else:
                action.append(np.argmax(np.array(pi)))

        return action

    def perform(self, policy_type='default'):
        ob = self.env.reset()
        # note this done is pre-decision to reset LSTM states!
        done = True
        self.model.reset()
        rewards = []
        while True:
            # policy-based on-poicy learning
            policy = self.model.forward(ob, done, 'p')
            self.env.update_fingerprint(policy)
            action = []
            for pi in policy:
                if policy_type != 'deterministic':
                    action.append(np.random.choice(
                        np.arange(len(pi)), p=pi))
                else:
                    action.append(np.argmax(np.array(pi)))

            next_ob, reward, done, global_reward = self.env.step(action)
            rewards.append(global_reward)
            if done:
                break
            ob = next_ob
        mean_reward = np.mean(np.array(rewards))
        std_reward = np.std(np.array(rewards))
        return mean_reward, std_reward

    def run_thread(self, coord):
        '''Multi-threading is disabled'''
        ob = self.env.reset()
        done = False
        cum_reward = 0
        while not coord.should_stop():
            ob, done, R, cum_reward = self.explore(ob, done, cum_reward)
            global_step = self.global_counter.cur_step
            if self.agent.endswith('a2c'):
                self.model.backward(R, self.summary_writer, global_step)
            else:
                self.model.backward(self.summary_writer, global_step)
            self.summary_writer.flush()
            if (self.global_counter.should_stop()) and (not coord.should_stop()):
                self.env.terminate()
                coord.request_stop()
                logging.info('Training: stop condition reached!')
                return

    def run(self):
        while not self.global_counter.should_stop():
            # test
            if self.run_test and self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                self.env.train_mode = False
                for test_ind in range(self.test_num):
                    mean_reward, std_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(mean_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'avg_reward': mean_reward,
                           'std_reward': std_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step, is_train=False)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
            # train
            self.env.train_mode = True
            ob = self.env.reset()
            # note this done is pre-decision to reset LSTM states!
            done = True
            self.model.reset()
            self.cur_step = 0
            rewards = []
            while True:
                ob, done, R, cur_rewards = self.explore(ob, done)
                rewards += cur_rewards
                global_step = self.global_counter.cur_step
                if self.agent.endswith('a2c'):
                    self.model.backward(R, self.summary_writer, global_step)
                else:
                    self.model.backward(self.summary_writer, global_step)
                # termination
                if done:
                    # self.env.terminate()
                    break
            rewards = np.array(rewards)
            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            log = {'agent': self.agent,
                   'step': global_step,
                   'test_id': -1,
                   'avg_reward': mean_reward,
                   'std_reward': std_reward}
            self.data.append(log)
            self._add_summary(mean_reward, global_step)
            self.summary_writer.flush()
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Tester(Trainer):
    def __init__(self, env, model, global_counter, summary_writer, output_path):
        super().__init__(env, model, global_counter, summary_writer)
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.data = []
        logging.info('Testing: total test num: %d' % self.test_num)

    def _init_summary(self):
        self.reward = tf.placeholder(tensorflow.float32, [])
        self.summary = tf.summary.scalar('test_reward', self.reward)

    def run_offline(self):
        # enable traffic measurments for offline test
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        rewards = []
        for test_ind in range(self.test_num):
            rewards.append(self.perform(test_ind))
            self.env.terminate()
            time.sleep(2)
            self.env.collect_tripinfo()
        avg_reward = np.mean(np.array(rewards))
        logging.info('Offline testing: avg R: %.2f' % avg_reward)
        self.env.output_data()

    def run_online(self, coord):
        self.env.cur_episode = 0
        while not coord.should_stop():
            time.sleep(30)
            if self.global_counter.should_test():
                rewards = []
                global_step = self.global_counter.cur_step
                for test_ind in range(self.test_num):
                    cur_reward = self.perform(test_ind)
                    self.env.terminate()
                    rewards.append(cur_reward)
                    log = {'agent': self.agent,
                           'step': global_step,
                           'test_id': test_ind,
                           'reward': cur_reward}
                    self.data.append(log)
                avg_reward = np.mean(np.array(rewards))
                self._add_summary(avg_reward, global_step)
                logging.info('Testing: global step %d, avg R: %.2f' %
                             (global_step, avg_reward))
                # self.global_counter.update_test(avg_reward)
        df = pd.DataFrame(self.data)
        df.to_csv(self.output_path + 'train_reward.csv')


class Predictor(Tester):
    def __init__(self, env, model, demo=False, policy_type='default'):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.demo = demo
        self.policy_type = policy_type

    def run(self, state):
        self.env.cur_episode = 0
        time.sleep(1)
        for test_ind in range(self.test_num):
            action = self.inference(state, policy_type=self.policy_type)
            time.sleep(2)

        return action 


class Evaluator(Tester):
    def __init__(self, env, model, output_path, demo=False, policy_type='default'):
        self.env = env
        self.model = model
        self.agent = self.env.agent
        self.env.train_mode = False
        self.test_num = self.env.test_num
        self.output_path = output_path
        self.demo = demo
        self.policy_type = policy_type

    def run(self):
        is_record = True
        record_stats = False
        self.env.cur_episode = 0
        self.env.init_data(is_record, record_stats, self.output_path)
        time.sleep(1)
        for test_ind in range(self.test_num):
            reward, _ = self.perform(policy_type=self.policy_type)
            logging.info('test %i, avg reward %.2f' % (test_ind, reward))
            time.sleep(2)
        self.env.output_data()
