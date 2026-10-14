import math
import random


def objective_function(x):
    value = 125.0 * math.exp(-0.1 * x) * math.sin(x)
    return value


def initialize_population(pop_size, lower_bound, upper_bound):
    population = [random.uniform(lower_bound, upper_bound) for _ in range(pop_size)]
    return population


def evaluate_population(population):
    fitnesses = [objective_function(x) for x in population]
    return fitnesses


def tournament_selection(population, fitnesses, tournament_size):
    selected_indices = random.sample(range(len(population)), tournament_size)
    best_index = selected_indices[0]
    for idx in selected_indices[1:]:
        if fitnesses[idx] > fitnesses[best_index]:
            best_index = idx
    parent = population[best_index]
    return parent


def arithmetic_crossover(parent1, parent2, crossover_rate, lower_bound, upper_bound):
    if random.random() >= crossover_rate:
        return parent1, parent2
    alpha = random.random()
    child1 = alpha * parent1 + (1.0 - alpha) * parent2
    child2 = alpha * parent2 + (1.0 - alpha) * parent1
    child1 = max(lower_bound, min(upper_bound, child1))
    child2 = max(lower_bound, min(upper_bound, child2))
    return child1, child2


def gaussian_mutation(individual, mutation_rate, lower_bound, upper_bound, sigma):
    if random.random() < mutation_rate:
        mutated = individual + random.gauss(0.0, sigma)
        mutated = max(lower_bound, min(upper_bound, mutated))
        return mutated
    return individual


def run_genetic_algorithm(
    population_size,
    generations,
    lower_bound,
    upper_bound,
    crossover_rate,
    mutation_rate,
    tournament_size,
    mutation_sigma,
):
    population = initialize_population(population_size, lower_bound, upper_bound)
    best_history = []
    populations_history = []
    best_x = population[0]
    best_f = objective_function(best_x)
    for _ in range(generations):
        populations_history.append(list(population))
        fitnesses = evaluate_population(population)
        for idx, fit in enumerate(fitnesses):
            if fit > best_f:
                best_f = fit
                best_x = population[idx]
        best_history.append(best_f)
        next_population = []
        while len(next_population) < population_size:
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)
            child1, child2 = arithmetic_crossover(
                parent1, parent2, crossover_rate, lower_bound, upper_bound
            )
            child1 = gaussian_mutation(
                child1, mutation_rate, lower_bound, upper_bound, mutation_sigma
            )
            child2 = gaussian_mutation(
                child2, mutation_rate, lower_bound, upper_bound, mutation_sigma
            )
            next_population.append(child1)
            if len(next_population) < population_size:
                next_population.append(child2)
        population = next_population
    return best_x, best_f, best_history, populations_history


def plot_results(best_x, best_f, best_history, populations_history, lower_bound, upper_bound):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib 不可用，改为文本输出")
        print("前 5 代最优值：", [round(v, 6) for v in best_history[:5]])
        print("后 5 代最优值：", [round(v, 6) for v in best_history[-5:]])
        return None
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    xs = [lower_bound + i * (upper_bound - lower_bound) / 500.0 for i in range(501)]
    ys = [objective_function(x) for x in xs]
    plt.figure(figsize=(10, 4))
    plt.plot(best_history, color="tab:blue")
    plt.title("收敛曲线")
    plt.xlabel("代数")
    plt.ylabel("最优值")
    plt.tight_layout()
    plt.figure(figsize=(10, 4))
    plt.plot(xs, ys, color="tab:green")
    plt.scatter([best_x], [best_f], color="red")
    plt.title("目标函数与最优解")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.tight_layout()
    plt.figure(figsize=(10, 4))
    for gen_index, population in enumerate(populations_history):
        if gen_index % 5 == 0:
            xs_scatter = [gen_index] * len(population)
            plt.scatter(xs_scatter, population, s=8, alpha=0.5, color="tab:purple")
    plt.title("种群分布演化")
    plt.xlabel("代数")
    plt.ylabel("个体位置 x")
    plt.tight_layout()
    plt.show()
    return None







def main():
    # 1.参数设置与初始化
    population_size = 60
    generations = 120
    lower_bound = -2.0
    upper_bound = 2.0
    crossover_rate = 0.85
    mutation_rate = 0.2
    tournament_size = 3
    mutation_sigma = 0.2
    # 2.运行遗传算法并输出结果
    best_x, best_f, best_history, populations_history = run_genetic_algorithm(
        population_size,
        generations,
        lower_bound,
        upper_bound,
        crossover_rate,
        mutation_rate,
        tournament_size,
        mutation_sigma,
    )
    # 3.生成实验分析报告
    print("最优解 x* = {:.6f}".format(best_x))
    print("最优值 f(x*) = {:.6f}".format(best_f))
    print("最后一代最优值 = {:.6f}".format(best_history[-1]))
    print("参数：pop={}, gen={}, pc={}, pm={}, tour={}, sigma={}".format(
        population_size,
        generations,
        crossover_rate,
        mutation_rate,
        tournament_size,
        mutation_sigma,
    ))
    plot_results(
        best_x,
        best_f,
        best_history,
        populations_history,
        lower_bound,
        upper_bound,
    )


if __name__ == "__main__":
    random.seed(2026)
    main()
