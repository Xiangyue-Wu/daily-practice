using Random

# 1. 数据初始化
weights = [2, 2, 6, 5, 4, 3]
values  = [6, 3, 5, 4, 6, 5]
capacity = 10
n_items = length(weights)

# 2. 参数设置
pop_size = 50
max_gen = 100
pc = 0.8
pm = 0.05
elite_count = 2  # 每代强制保留最好的2个个体

# 3. 初始化种群
# 设置随机种子以保证结果可复现 
Random.seed!(42) 
population = rand(0:1, pop_size, n_items)

# 4. 适应度计算函数
function calculate_fitness(pop, w, v, C)
    fits = zeros(size(pop, 1))
    for i in 1:size(pop, 1)
        ind = pop[i, :]
        total_w = sum(ind .* w)
        total_v = sum(ind .* v)
        
        # 硬约束惩罚：超重直接为0
        if total_w <= C
            fits[i] = Float64(total_v)
        else
            fits[i] = 0.0
        end
    end
    return fits
end

# 5. 选择操作 (锦标赛)
function selection(pop, fits)
    new_pop = similar(pop)
    n = size(pop, 1)
    for i in 1:n
        candidates = rand(1:n, 3) # 竞赛规模 k=3
        best_idx = candidates[1]
        for k in 2:3
            if fits[candidates[k]] > fits[best_idx]
                best_idx = candidates[k]
            end
        end
        new_pop[i, :] = pop[best_idx, :]
    end
    return new_pop
end

# 6. 交叉操作 (单点交叉)
function crossover(pop, pc)
    n_rows, n_cols = size(pop)
    limit = iseven(n_rows) ? n_rows : n_rows - 1
    for i in 1:2:limit
        if rand() < pc
            point = rand(1:(n_cols-1))
            # 交换基因片段
            temp = copy(pop[i, point+1:end])
            pop[i, point+1:end] = pop[i+1, point+1:end]
            pop[i+1, point+1:end] = temp
        end
    end
    return pop
end

# 7. 变异操作
function mutation(pop, pm)
    n_rows, n_cols = size(pop)
    for i in 1:n_rows
        for j in 1:n_cols
            if rand() < pm
                pop[i, j] = 1 - pop[i, j]
            end
        end
    end
    return pop
end

# 8. 主循环
println("=== 0-1 背包问题遗传算法求解 ===")
println("物品数量: $n_items, 背包容量: $capacity")
println("参数: 种群=$pop_size, 迭代=$max_gen, Pc=$pc, Pm=$pm, 精英数=$elite_count")
println("--------------------------------")

best_history = []
# 正确初始化全局变量
global_best_val = 0.0
global_best_sol = zeros(Int, n_items)

for gen in 1:max_gen
    # 1. 计算适应度
    fits = calculate_fitness(population, weights, values, capacity)
    
    # 2. 记录当前代最优
    current_max_val = maximum(fits)
    current_best_idx = argmax(fits)
    current_best_sol = population[current_best_idx, :]
    
    # 明确声明要修改全局变量
    global global_best_val, global_best_sol
    if current_max_val > global_best_val
        global_best_val = current_max_val
        global_best_sol = copy(current_best_sol)
    end
    
    push!(best_history, current_max_val)
    
    # 打印进度 (每20代或首尾)
    if gen == 1 || gen % 20 == 0 || gen == max_gen
        w_curr = sum(current_best_sol .* weights)
        println("代数: $gen | 当代最优价值: $current_max_val | 重量: $w_curr | 方案: $current_best_sol")
    end
    
    # 精英保留策略
    # 找出当前种群中适应度最高的 elite_count 个个体
    sorted_indices = sortperm(fits, rev=true)
    elites = population[sorted_indices[1:elite_count], :]
    
    # 选择 (生成除去精英位置的剩余种群)
    selected_pop = selection(population, fits)
    crossed_pop = crossover(selected_pop, pc)
    new_population = mutation(crossed_pop, pm)
    
    # 计算新种群的适应度，找出最差的几个位置替换为精英
    # 将精英个体放入新种群的前 elite_count 行
    for k in 1:elite_count
        new_population[k, :] = elites[k, :]
    end
    
    # 明确声明要修改全局的 population 变量
    global population = new_population
end

# 9. 最终结果验证与输出
final_fits = calculate_fitness(population, weights, values, capacity)
final_best_idx = argmax(final_fits)
final_sol = population[final_best_idx, :]
max_value = final_fits[final_best_idx]
total_weight = sum(final_sol .* weights)

println("--------------------------------")
println("=== 最终优化结果 ===")
println("最优选择向量 (1=选中): ", final_sol)
selected_ids = findall(x -> x == 1, final_sol)
println("选中的物品编号: ", selected_ids)
println("最大总价值: ", max_value)
println("实际总重量: ", total_weight)
println("是否满足容量限制: ", total_weight <= capacity ? "是" : "否")
println("理论最优解验证: ", (max_value == 17 && total_weight <= 10) ? "成功找到理论最优" : "未找到理论最优，请检查参数")