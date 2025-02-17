"""Synthetic benchmark for comparing end-to-end latencies in ROS 2 scheduling.

This module implements a synthetic benchmark to evaluate and compare the end-to-end
latencies of task chains under Rate Monotonic (RM) and default ROS 2 scheduling
policies. It generates synthetic task sets and chains, calculates response times
and end-to-end latencies, and provides visualization of the results.

The benchmark includes:
- Generation of synthetic task sets with specified utilization levels
- Creation of cause-effect chains from the task sets
- Calculation of response times using Rate Monotonic analysis
- Calculation of end-to-end latencies for both RM and default ROS 2 scheduling
- Statistical analysis of the results including mean, standard deviation,
    percentiles, and confidence intervals
- Visualization of the results through histograms

Functions:
        evaluation(utilization): Evaluates end-to-end latency for a given utilization
        calculate_end_to_end_latency(task_set): Calculates latency for ROS 2 scheduling
        calculate_response_times(taskset): Calculates response times using RM analysis
        get_max_blocking_from_lower_priority(taskset, task_index): Calculates blocking time
        calculate_higher_priority_interference(taskset, task_index, interval): Calculates interference
        plot_results(tables): Creates visualization of benchmark results
        main(): Runs the benchmark for different utilization levels

Classes:
        EvalTask: Represents a task in the evaluation process
        Chain: Represents a chain of tasks in the evaluation process"""

import math
import matplotlib.pyplot as plt
from waters.benchmark_WATERS import gen_taskset, gen_ce_chains
import argparse

class EvalTask:
    """Represents a task in the evaluation process.

    This class stores a task and its calculated response times and
    end-to-end latencies under different scheduling policies.
    """
    def __init__(self, task):
        self.task = task
        self.rm_response_time = 0
        self.rm_end_to_end_latency = 0
        self.default_end_to_end_latency = 0

class Chain:
    """Represents a chain of tasks in the evaluation process.

    This class stores a chain of tasks and its calculated end-to-end
    latencies under different scheduling policies.
    """
    def __init__(self, chain):
        self.chain = chain
        self.rm_end_to_end_latency = 0
        self.default_end_to_end_latency = 0

def evaluation(utilization, num_task_sets=1000):
    """Evaluates the end-to-end latency of task chains under RM and default ROS 2 scheduling.

    This function generates task sets and chains, calculates response times and end-to-end latencies,
    and returns a table of results.

    Args:
        utilization (float): The utilization level for generating task sets.
        num_task_sets (int): The number of task sets to generate. Default is 1000.
    """
    print("Creating Task Sets")
    ts_set = [gen_taskset(utilization) for _ in range(num_task_sets)]
    ce_set = [gen_ce_chains(ts) for ts in ts_set]

    # Transform the task set
    eval_task_sets = []
    for ts in ts_set:
        eval_tasks = [EvalTask(task) for task in ts]
        eval_task_sets.append(eval_tasks)

    # Sort the task set in eval_task_sets by period
    eval_task_sets = [sorted(ts, key=lambda x: x.task.rel.period) for ts in eval_task_sets]

    # Determine the minimum and maximum number of tasks per task set
    print("Minimum Number of Tasks: ", min(len(ts) for ts in eval_task_sets))
    print("Maximum Number of Tasks: ", max(len(ts) for ts in eval_task_sets))

    # Transform the chains
    eval_chain_set = []
    for ce in ce_set:
        eval_chain = []
        for chain in ce:
            task_chain = [next(task for task in eval_task_sets[ce_set.index(ce)] if task.task == chain_task) for chain_task in chain]
            eval_chain.append(Chain(task_chain))
        eval_chain_set.append(eval_chain)

    # Determine the minimum and maximum number of chains per chain set, and the minimum and maximum number of tasks per chain
    print("Minimum Number of Chains: ", min(len(ce) for ce in eval_chain_set))
    print("Maximum Number of Chains: ", max(len(ce) for ce in eval_chain_set))
    print("Minimum Number of Tasks per Chain: ", min(len(chain.chain) for ce in eval_chain_set for chain in ce))
    print("Maximum Number of Tasks per Chain: ", max(len(chain.chain) for ce in eval_chain_set for chain in ce))

    # Calculate utilization of each task set
    for eval_ts in eval_task_sets:
        utilization = sum(task.task.ex.wcet / task.task.rel.period for task in eval_ts)
        # print("Utilization: ", utilization)

    for eval_ts in eval_task_sets:
        calculate_response_times(eval_ts)

    indices_to_remove = []
    number_schedulable = 0
    number_not_schedulable = 0
    for eval_ts in eval_task_sets:
        schedulable = all(task.rm_response_time <= task.task.rel.period for task in eval_ts)
        if schedulable:
            number_schedulable += 1
        else:
            number_not_schedulable += 1
            indices_to_remove.append(eval_task_sets.index(eval_ts))

    print("Number of Schedulable Task Sets: ", number_schedulable)
    print("Number of Not Schedulable Task Sets: ", number_not_schedulable)

    for index in sorted(indices_to_remove, reverse=True):
        # print("Removing Task Set: ", index)
        del eval_task_sets[index]
        del eval_chain_set[index]

    for eval_chains in eval_chain_set:
        # print("Number of Chains: ", len(eval_chains))
        for eval_chain in eval_chains:
            # print("Chain Length: ", len(eval_chain.chain))
            chain_end_to_end_latency = sum(task.task.rel.period + task.rm_response_time for task in eval_chain.chain)
            # print("End-to-End Latency: ", chain_end_to_end_latency)
            eval_chain.rm_end_to_end_latency = chain_end_to_end_latency

    for eval_ts in eval_task_sets:
        calculate_end_to_end_latency(eval_ts)

    for eval_chains in eval_chain_set:
        # print("Number of Chains: ", len(eval_chains))
        for eval_chain in eval_chains:
            # print("Chain Length: ", len(eval_chain.chain))
            chain_end_to_end_latency = sum(task.default_end_to_end_latency for task in eval_chain.chain)
            # print("End-to-End Latency: ", chain_end_to_end_latency)
            eval_chain.default_end_to_end_latency = chain_end_to_end_latency

    print("===Printing Results===")

    # Print a comparison of the results
    table = []
    for i, eval_chains in enumerate(eval_chain_set):
        table.extend(
            [
                i,
                j,
                eval_chain.rm_end_to_end_latency,
                eval_chain.default_end_to_end_latency,
                eval_chain.rm_end_to_end_latency - eval_chain.default_end_to_end_latency,
                (eval_chain.default_end_to_end_latency - eval_chain.rm_end_to_end_latency) / eval_chain.default_end_to_end_latency * 100,
            ]
            for j, eval_chain in enumerate(eval_chains)
        )

    # Calculate and print statistics for RM end-to-end latency
    print_statistics(table, 2, "RM End-to-End Latency Values")

    # Calculate and print statistics for default end-to-end latency
    print_statistics(table, 3, "ROS 2 Default End-to-End Latency Values")

    # Calculate and print statistics for the difference in percentage
    print_statistics(table, 5, "Difference Values")

    # Plot histogram of percentage (between 0 and 1 with 200 bins)
    plt.hist([row[5] for row in table], bins=200, density=False)
    plt.xlabel("Normalized Reduction %")
    plt.ylabel("Percentage (%)")
    plt.title("Normalized Reduction of End-to-End Latency")
    plt.xlim(-15, 105)
    plt.xticks(range(-10, 110, 10))
    plt.show()

    return table

def calculate_end_to_end_latency(task_set):
    """Calculate end-to-end latency for each task in the taskset using the default ROS 2 scheduling policy."""
    executor_wcet_sum = sum(task.task.ex.wcet for task in task_set)

    for i, task in enumerate(task_set):
        higher_priority_wcet = sum(task.task.ex.wcet for task in task_set[:i])
        task_execution_time = task.task.ex.wcet
        task.default_end_to_end_latency = executor_wcet_sum + max(0, task.task.rel.period - task.task.ex.wcet + higher_priority_wcet) + task_execution_time

def calculate_response_times(taskset):
    """Calculate response times for each task in the taskset using Rate Monotonic analysis."""
    for i, task in enumerate(taskset):
        max_blocking_lower = get_max_blocking_from_lower_priority(taskset, i)
        response_time = task.task.ex.wcet + max_blocking_lower

        while True:
            interference = calculate_higher_priority_interference(taskset, i, response_time)
            new_response_time = task.task.ex.wcet + max_blocking_lower + interference

            if new_response_time == response_time:
                break
            if new_response_time > 100000000000:
                break

            response_time = new_response_time

        task.rm_response_time = response_time

def get_max_blocking_from_lower_priority(taskset, task_index):
    """Get maximum blocking time from lower priority tasks."""
    if task_index >= len(taskset) - 1:
        return 0
    return max(task.task.ex.wcet for task in taskset[task_index + 1:])

def calculate_higher_priority_interference(taskset, task_index, interval):
    """Calculate interference from higher priority tasks over given interval."""
    interference = 0
    for j in range(task_index):
        higher_task = taskset[j]
        n_activations = math.ceil(interval / higher_task.task.rel.period)
        interference += n_activations * higher_task.task.ex.wcet
    return interference

def plot_results(tables):
    """Plots the results of the benchmark.

    This function creates a histogram of the normalized reduction in end-to-end latency
    for each utilization level. It also calculates and prints descriptive statistics
    for the RM, ROS 2 default, and difference values.
    """
    fig, axs = plt.subplots(len(tables), 1, figsize=(10, 10))
    fig.tight_layout()

    for i, table in enumerate(tables):
        plt.rcParams.update({'font.size': 20})
        axs[i].tick_params(axis='both', which='major', labelsize=20)
        axs[i].set_xlabel('xlabel', fontsize=20)
        axs[i].set_ylabel('ylabel', fontsize=20)

        axs[i].hist([row[5] for row in table], bins=200, density=False)
        if i == len(tables) - 1:
            axs[i].set_xlabel("Normalized Reduction %")
        else:
            axs[i].set_xlabel("")
        axs[i].set_ylabel("Count")
        if i == 0:
            axs[i].set_title("Normalized Reduction of End-to-End Latency")

        axs[i].text(
            1.05,
            0.5,
            f"Utilization {str([0.6, 0.8, 0.9][i])}",
            horizontalalignment='center',
            verticalalignment='center',
            rotation=90,
            transform=axs[i].transAxes,
        )

        axs[i].set_xlim(-15, 105)
        axs[i].set_xticks(range(-10, 110, 10))
        plt.tight_layout()

        print_statistics(table, 2, "RM End-to-End Latency Values")
        print_statistics(table, 3, "ROS 2 Default End-to-End Latency Values")
        print_statistics(table, 5, "Difference Values")

    plt.savefig("evaluation_plot.png")
    plt.show()

def print_statistics(table, column_index, title):
    """Prints statistical analysis for a given column in the table."""
    mean = sum(row[column_index] for row in table) / len(table)
    std = math.sqrt(sum((row[column_index] - mean) ** 2 for row in table) / len(table))
    min_diff = min(row[column_index] for row in table)
    max_diff = max(row[column_index] for row in table)
    percentile_99 = sorted([row[column_index] for row in table])[int(len(table) * 0.99)]
    n = len(table)
    z = 1.96
    lower = mean - z * (std / math.sqrt(n))
    upper = mean + z * (std / math.sqrt(n))

    print(title)
    print("Mean: ", mean)
    print("Standard Deviation: ", std)
    print("Min Difference: ", min_diff)
    print("Max Difference: ", max_diff)
    print("99th Percentile: ", percentile_99)
    print("Confidence Interval: ", lower, upper)

def main():
    """Runs the main evaluation and plotting process.

    This function performs the benchmark evaluation for different utilization levels
    and then plots the collected results.
    """
    parser = argparse.ArgumentParser(description="Run the synthetic benchmark for ROS 2 scheduling.")
    parser.add_argument('--num_task_sets', type=int, default=1000, help='Number of task sets to generate')
    args = parser.parse_args()

    table_60 = evaluation(0.6, num_task_sets=args.num_task_sets)
    table_80 = evaluation(0.8, num_task_sets=args.num_task_sets)
    table_90 = evaluation(0.9, num_task_sets=args.num_task_sets)

    plot_results([table_60, table_80, table_90])

if __name__ == "__main__":
    main()
