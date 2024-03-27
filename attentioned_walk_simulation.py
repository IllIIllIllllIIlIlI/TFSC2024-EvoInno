import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as stat
from functools import partial
from collections import namedtuple
from pprint import pprint
from random import sample
import string
from datetime import datetime
from pathlib import Path

from csv import writer as csv_writer
from csv import reader as csv_reader

# get saving directory and make sure it exists
exec_path = Path(__file__).parent
active_path = exec_path / 'runs'
active_path.mkdir(exist_ok=True)

# color for attention
attention_c = plt.get_cmap('viridis')(0.7)


char_set = string.ascii_uppercase + string.digits


TEntry = namedtuple('TEntry', ['perceived_f', 'underlying_f',
                                        'update_vec', 'starting_vec', 'ending_vec',
                                        'starting_dist', 'ending_dist',
                                        'status'])

AEntry = namedtuple('AEntry', ['attention_balance', 'attention_vec'])


class Experiment(object):
    """
    Contains the setting in which the exploration
    is happening

    """

    def __init__(self, latent_dims, latent_dims_scale_distro,
                 inherent_noise_distro, inherent_noise_scale=1,
                 local_noise_distro=None, local_noise_scale=1):
        """


        :param latent_dims:
        :param latent_dims_scale_distro:
        :param inherent_noise_distro:
        :param inherent_noise_scale:
        :param local_noise_distro:
        :param local_noise_scale:
        """

        self.dims = latent_dims
        self.dims_width_distro = latent_dims_scale_distro
        self.inherent_noise_distro = inherent_noise_distro
        self.inherent_noise_scale = inherent_noise_scale
        self.local_noise_distro = local_noise_distro
        self.local_noise_scale = local_noise_scale

        self.fitness_f = None

        self.testing_trace = []
        self.attention_trace = []

        self.random_tag = ''.join(sample(char_set * 10, 10))

    def generate_fitness_landscape(self):

        def fitness_f(vector):

            noise = self.inherent_noise_distro(size=1)
            fitness = part_norm(vector)/corr_factor + noise

            return fitness

        dim_widths = self.dims_width_distro(size=self.dims)
        width_mat = np.diag(np.abs(dim_widths))
        part_norm = stat.multivariate_normal(cov=width_mat).pdf
        corr_factor = part_norm(np.zeros((self.dims,)))

        fitness_f = np.vectorize(fitness_f, otypes=[float], signature='(n)->()')
        self.fitness_f = fitness_f

        return fitness_f

    def visualize_fitness_f(self, render=True):
        """
        Mostly a 2-D debug function.

        :return:
        """

        if self.dims > 2:
            raise Exception("Too many dims to visualize!")
        else:
            x, y = np.mgrid[-1:1:.01, -1:1:.01]
            pos = np.dstack((x, y))

            plt.contourf(x, y, self.fitness_f(pos))

            if render:
                plt.show()


    def try_step(self, start_v, length):
        """
        Wrapper for a single tick of a search for a better solution

        :param start_v:
        :param length:
        :return:
        """
        step = np.random.normal(0.0, length, size=(self.dims,))

        tested_location = start_v + step
        underlying_fitness = self.fitness_f(tested_location)
        if self.local_noise_distro is not None:
            perceived_fitness = underlying_fitness + \
                                self.local_noise_scale * self.local_noise_distro(size=(self.dims,))
        else:
            perceived_fitness = underlying_fitness

        trace_e = TEntry(perceived_fitness, underlying_fitness,
                                step.tolist(), start_v.tolist(), tested_location.tolist(),
                                np.linalg.norm(start_v), np.linalg.norm(tested_location),
                                'rejected')

        self.testing_trace.append(trace_e)

        return perceived_fitness, step


    def try_step_with_attention(self, start_v, length, current_attention,
                                     oblivious_dimensions=None):
        """
        Wrapper for a single tick of a search for a better solution

        :param start_v:
        :param length:
        :param oblivious_dimensions: set of 1/0s indicating which dimensions are "incomprehensible"
        :return:
        """

        random_step = np.random.normal(0.0, length, size=(self.dims,))

        if oblivious_dimensions is not None:
            random_step = random_step * oblivious_dimensions

        if current_attention.attention_balance > 1:
            step = current_attention.attention_vec * current_attention.attention_balance
        else:
            step = current_attention.attention_vec * current_attention.attention_balance + \
                   random_step * (1 - current_attention.attention_balance)

        tested_location = start_v + step
        underlying_fitness = self.fitness_f(tested_location)
        if self.local_noise_distro is not None:
            perceived_fitness = underlying_fitness + \
                                self.local_noise_scale * self.local_noise_distro(size=(self.dims,))
        else:
            perceived_fitness = underlying_fitness

        trace_e = TEntry(perceived_fitness, underlying_fitness,
                         step.tolist(), start_v.tolist(), tested_location.tolist(),
                         np.linalg.norm(start_v), np.linalg.norm(tested_location),
                         'rejected')

        self.testing_trace.append(trace_e)

        return perceived_fitness, step


    def visualize_step(self, start_v, step_v, render=True):
        """
        Mostly a 2-D debug function.

        :return:
        """
        if self.dims > 2:
            raise Exception("Too many dims to visualize!")
        else:
            x, y = start_v
            dx, dy = step_v
            plt.arrow(x, y, dx, dy, color='r', width=0.01)

        if render:
            plt.show()

    def reset_walk(self, intensity=1.0):
        """
        Basically, creates the opportunity for improvement. It needs to still remain close to the
        optimum to fulfill the FGM assumptions and limit value theorems

        :return:
        """
        starting_v = np.random.normal(0, intensity / np.sqrt(self.dims), size=(self.dims,))

        if self.dims == 2:  # compensate for excessive variance in low dim
            starting_v = starting_v / np.linalg.norm(starting_v) * intensity


        starting_fitness = self.fitness_f(starting_v)

        trace_e = TEntry(starting_fitness, starting_fitness,
                         (starting_v*0).tolist(), starting_v.tolist(), starting_v.tolist(),
                          np.linalg.norm(starting_v), np.linalg.norm(starting_v),
                         'start')

        self.testing_trace.append(trace_e)

        return starting_fitness, starting_v

    def walk_blind(self, expire=10, step_length=0.25):
        """
        returns fitness and steps in case of attention-free exploration
        only some dimension can be touched, optionally (inherent limitations)

        :return:
        """

        i = expire
        unsuccessful_tests = 0

        current_f, current_v = self.reset_walk()

        while unsuccessful_tests < expire:
            perceived_new_fitness, tried_step = self.try_step(current_v, length=step_length)
            buffer_record = list(self.testing_trace[-1])

            if perceived_new_fitness > current_f:
                current_f = perceived_new_fitness
                current_v += tried_step
                buffer_record[-1] = 'accepted'
                i -= 1
                unsuccessful_tests = 0
            else:
                unsuccessful_tests += 1

            self.testing_trace[-1] = TEntry(*buffer_record)

    def visualize_trace(self):

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        fig.set_size_inches(7.5, 7.5, forward=True)
        fig.set_dpi(200)

        x, y = np.mgrid[-2:2:.01, -2:2:.01]
        pos = np.dstack((x, y))
        ax1.contourf(x, y, self.fitness_f(pos))

        ax3 = ax2.twinx()

        pprint(self.testing_trace)

        step_count = 0

        ax2.plot(0, self.testing_trace[0].perceived_f, marker=".", c='r')

        underlying_fitness_stack = [(0, self.testing_trace[0].underlying_f)]
        attention_stack = []

        for step, attention in zip(self.testing_trace, self.attention_trace):
            x, y = tuple(step.starting_vec)
            dx, dy = tuple(step.update_vec)

            step_count += 1

            attention_stack.append((step_count, attention.attention_balance))

            if step.status == 'accepted':
                _color = 'r'
                _color2 = 'b'

                underlying_fitness_stack.append((step_count, step.underlying_f))
                alpha = 0.75

            else:
                _color = 'k'
                _color2 = 'g'
                alpha = 0.15

            ax1.arrow(x, y, dx, dy, color=_color, width=0.01, alpha=alpha, length_includes_head=True)
            ax2.plot(step_count, step.perceived_f, marker=".", c=_color)

        x, y = tuple(self.testing_trace[-1].starting_vec)  # ending for last accept, starting for
        # rej

        ax1.plot(x, y, marker="*", c='r')

        ax1.set_xlim((-2, 2))
        ax1.set_ylim((-2, 2))

        ax1.set_title('Feature Space Search')

        ax2.set_title('Fitness and Attention Traces')

        underlying_fitness_stack = np.array(underlying_fitness_stack)
        attention_stack = np.array(attention_stack)

        ax2.plot(underlying_fitness_stack.T[0],
                 underlying_fitness_stack.T[1], c='r', label='adopted innovations')
        ax3.plot(attention_stack.T[0],
                 attention_stack.T[1], c=attention_c, label='attention')

        labels_rail = [('.', 'k'), ('_', 'r'), ('_', attention_c)]
        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
        handles = [f(*labels_rail[i]) for i in range(3)]
        labels =['failed iterations', 'successful iterations', 'attention']

        ax2.set_ylabel('technological system fitness')
        ax2.set_xlabel('improvement attempts')

        ax3.set_ylabel('attention', color=attention_c)
        ax3.tick_params(axis='y', labelcolor=attention_c)

        ax2.legend(handles, labels, framealpha=1, loc='lower right')

        ax2.set_ylim(bottom=0.0)
        ax3.set_ylim(bottom=0.0)

        plt.tight_layout()

        plt.show()

    def time_based_visualization_with_attention(self):

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        fig.set_size_inches(7.5, 7.5, forward=True)
        fig.set_dpi(100)

        x, y = np.mgrid[-2:2:.01, -2:2:.01]
        pos = np.dstack((x, y))
        ax1.contourf(x, y, self.fitness_f(pos))

        pprint(self.testing_trace)

        x, y = tuple(self.testing_trace[0].starting_vec)
        ax1.plot(x, y, marker="*", c='b')

        step_count = 0

        ax2.plot(0, self.testing_trace[0].underlying_f, marker=".", c='r')

        for step in self.testing_trace:
            x, y = tuple(step.starting_vec)
            dx, dy = tuple(step.update_vec)

            if step.status == 'accepted':
                _color = 'r'
                _color2 = 'b'
                xt, yt = tuple(step.ending_vec)
                ax1.plot(xt, yt, marker="*", c='w')
                step_count += 1
                ax2.plot(step_count, step.perceived_f, marker=".", c=_color)

            else:
                _color = 'k'
                _color2 = 'g'

            ax1.arrow(x, y, dx, dy, color=_color, width=0.01, alpha=0.25, length_includes_head=True)

        x, y = tuple(self.testing_trace[-1].starting_vec)  # ending for last accept, starting for
        # rej

        ax1.plot(x, y, marker="*", c='r')

        plt.show()

    def walk_continuous_attention(self):
        pass


    def walk_short_span_attention(self,
                                  expire=10,
                                  step_length=0.25,
                                  resets=None,  #step, intensity
                                  ):

        def attention_converter(old_AEntry, tried_step,
                                new_fitness, old_fitness):

            if (new_fitness - old_fitness) / max(old_fitness, 0.01) > 0.05:
                attention_balance = (old_AEntry.attention_balance +
                                     np.abs(np.dot(old_AEntry.attention_vec, tried_step))
                                     * ((new_fitness - old_fitness) / max(old_fitness, 0.01))
                                     / 0.05) / 2

                attention_vector = (old_AEntry.attention_vec * attention_balance
                                    + tried_step * (1 - attention_balance)) / 2


                current_attention = AEntry(attention_balance, attention_vector)
                self.attention_trace.append(current_attention)
                print('^ attention reinforced')


            else:
                attention_vector = old_AEntry.attention_vec
                attention_balance = old_AEntry.attention_balance / max(1.2,
                                        ((old_fitness - new_fitness) / max(old_fitness, 0.01)) /
                                                                       0.05)
                current_attention = AEntry(attention_balance, attention_vector)
                self.attention_trace.append(current_attention)


            print("attention modification debug:\n"
                  "\t vec_alignment: %.2f\n"
                  "\t fitness_amplifier: %.2f\n"
                  "\t trimmed_fitness_amplifier: %.2f\n"
                  "\t old_attention_balance: %.2f\n"
                  "\t new_attention_balance: %.2f" % (np.dot(old_AEntry.attention_vec,
                                                             tried_step),
                                                      ((new_fitness - old_fitness) / max(old_fitness, 0.01)) / 0.05,
                                                      1.0 / max(1.2, np.abs((new_fitness - old_fitness) / max(old_fitness, 0.01)) / 0.05),
                  old_AEntry.attention_balance,
                  current_attention.attention_balance)
                  )

            return current_attention


        total_ticks = 0
        unsuccessful_tests = 0

        reset_clock = {step: offset for step, offset in resets}

        random_attention = np.random.normal(0.0, step_length, size=(self.dims,))
        current_attention = AEntry(0.01, random_attention)
        self.attention_trace.append(current_attention)

        while unsuccessful_tests < expire:

            if reset_clock.get(total_ticks, False):
                print('environment change by %.2f at step %d' % (reset_clock[total_ticks],
                                                                 total_ticks))
                current_f, current_v = self.reset_walk(reset_clock[total_ticks])

            perceived_new_fitness, tried_step = self.try_step_with_attention(current_v,
                                                                             length=step_length,
                                                                             current_attention=current_attention)
            buffer_record = list(self.testing_trace[-1])

            current_attention = attention_converter(current_attention, tried_step,
                                                    perceived_new_fitness, current_f)

            if perceived_new_fitness > current_f:
                current_f = perceived_new_fitness
                current_v += tried_step
                buffer_record[-1] = 'accepted'
                unsuccessful_tests = 0

            else:
                unsuccessful_tests += 1

            self.testing_trace[-1] = TEntry(*buffer_record)

            print(self.testing_trace[-1])
            print(self.attention_trace[-1])
            print('>>')

            total_ticks += 1

    def register_experiment(self, assembly_key):

        def extract_underlying_attention_and_fitness():

            percieved_f_stack = [self.testing_trace[0].perceived_f]
            underlying_f_stack = [self.testing_trace[0].underlying_f]
            attention_stack = [0.0]

            for step, attention in zip(self.testing_trace, self.attention_trace):

                attention_stack.append(attention.attention_balance)
                percieved_f_stack.append(step.perceived_f)

                if step.status == 'accepted':
                    underlying_f_stack.append(step.underlying_f)
                else:
                    underlying_f_stack.append(np.nan)

            return percieved_f_stack, underlying_f_stack, attention_stack

        # get saving date
        root_reg = datetime.now().strftime("%Y-%m-%d-%Hh-%Mm-%Ss")

        # form the name payload:
        name = '_'.join([self.random_tag, root_reg, assembly_key]) + '.csv'
        save_path = active_path / name

        tristack = extract_underlying_attention_and_fitness()
        np_stack = np.vstack(tristack)
        np_stack = np_stack.T

        with open(save_path, 'wt') as dest:
            writer = csv_writer(dest)
            writer.writerows(np_stack)


def decompress_block(block):

    rail = [i for i in range(0, len(block[0]))]

    perceived_f = block[0]

    arr_underlying_f = np.array(block[1])

    second_rail = np.argwhere(np.logical_not(np.isnan(arr_underlying_f)))
    arr_underlying_f = arr_underlying_f[second_rail]

    second_rail = second_rail.tolist()
    underlying_f = arr_underlying_f.tolist()

    attention = block[2]

    return rail, second_rail, perceived_f, underlying_f, attention


def show_traces(experiment_traces=None, title='Fitness and Attention Traces'):

    def show_trace(block, alpha=1):

        rail, second_rail, perceived_f, underlying_f, attention = decompress_block(block)

        ax1.scatter(rail, perceived_f, marker=".", c='k', alpha=alpha)
        ax1.plot(second_rail, underlying_f, marker=".", c='r', alpha=alpha)
        ax1.plot(second_rail, underlying_f, c='r', alpha=alpha)
        ax2.plot(rail, attention, c=attention_c, alpha=alpha)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    fig.set_size_inches(6, 4, forward=True)
    fig.set_dpi(200)

    alphas = np.linspace(0.9, 1, len(experiment_traces)).tolist()

    for alpha, trace_block in zip(alphas, experiment_traces):
        show_trace(trace_block, alpha)

    ax1.set_title(title)

    labels_rail = [('.', 'k'), ('_', 'r'), ('_', attention_c)]
    f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f(*labels_rail[i]) for i in range(3)]
    labels = ['failed iterations', 'successful iterations', 'attention']

    ax1.set_ylabel('technological system fitness')
    ax1.set_xlabel('improvement attempts')

    ax2.set_ylabel('attention', color=attention_c)
    ax2.tick_params(axis='y', labelcolor=attention_c)

    ax1.legend(handles, labels, framealpha=1, loc='lower right')

    ax1.set_ylim(bottom=0.0)
    ax2.set_ylim(bottom=0.0)

    plt.tight_layout()

    plt.show()


def pull_experiments(assembly_key=None):

    exp_aggregation = []

    for subpath in active_path.iterdir():

        if subpath.stem.endswith(assembly_key):

            print(subpath)

            with open(subpath, 'rt') as source:
                reader = csv_reader(source)
                block = []
                for line in reader:
                    if len(line) > 0:
                        block.append(line)
                print(block)
                block = np.array(block).astype(np.float).T
                print(block.shape)
                print(block)
                exp_aggregation.append(block.tolist())

    return exp_aggregation


if __name__ == "__main__":

    # experiment_sequence = 'expiration_10'
    # experiment_sequence = 'expiration_30'
    # experiment_sequence = 'expiration_10_sl_005'
    # experiment_sequence = 'expiration_30_sl_005'
    # experiment_sequence = 'expiration_10_sl_01_sys_chocks'
    # experiment_sequence = 'expiration_30_sl_01_sys_chocks'
    experiment_sequence = 'expiration_30_sl_005_sys_chocks'


    replicates_to_generate = 0
    latent_dims = 2


    titles_map = {'expiration_10': 'Base Scenario',
    'expiration_30': 'Gov. Support',
    'expiration_10_sl_005': 'Smaller Companies',
    'expiration_30_sl_005': 'Gov. Support & Smaller Companies',
    'expiration_10_sl_01_sys_chocks': 'Rapidly Changing Environment',
    'expiration_30_sl_01_sys_chocks': 'Gov. Support & Rapidly Changing Environment',
    'expiration_30_sl_005_sys_chocks': 'Gov. Support & Smaller Companies\n& Rapidly Changing '
                                       'Environment',
                  }

    noise_distro = partial(np.random.normal, loc=0, scale=0.01)
    dims_width_disto = partial(np.random.normal, loc=0.25, scale=0.15)


    while replicates_to_generate > 0:

        experiment = Experiment(latent_dims=latent_dims,
                                latent_dims_scale_distro=dims_width_disto,
                                inherent_noise_distro=noise_distro)

        experiment.generate_fitness_landscape()

        experiment.walk_short_span_attention(expire=30,
                                             step_length=0.05,
                                             # resets=[(0, 1.5)],
                                             resets=[(0, 1.5), (40, 1.0), (80, 1.0)],
                                             )
        experiment.register_experiment(experiment_sequence)

        replicates_to_generate -= 1

    test_blocks = pull_experiments(experiment_sequence)

    show_traces(test_blocks, title=titles_map.get(experiment_sequence, experiment_sequence))
