import random
import neat
import os
import matplotlib.pyplot as plt
import seaborn as sns
import timeit
import statistics

# Set configuration file path
local_directory = os.path.dirname(__file__)
config_path = os.path.join(local_directory, "config-feedforward")

# Store time results, generations, and genetic diversity results
time_results = []
population_sizes = []
gen_counts = []
diversity_results = []

def get_config():
    """Returns NEAT config"""
    return neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )

def random_fitness(_genome, _config):
    """Assign random fitness to a genome"""
    return random.random()

def evaluate_population(genomes, config):
    """Evaluate the fitness of each genome"""
    for genome_id, genome in genomes:
        genome.fitness = random_fitness(genome, config)

def compute_genetic_diversity(genomes):
    """Compute genetic diversity based on fitness variance"""
    fitnesses = [genome.fitness for genome_id, genome in genomes if genome.fitness is not None]
    if len(fitnesses) > 1:
        return statistics.variance(fitnesses)
    return 0

def evolve_population(config, target_fitness, max_gens=1000):
    """Run the evolutionary process"""
    pop = neat.Population(config)
    
    best_genome_fitness = 0
    no_improvement = 0
    stop_after = 15  # generations without improvement
    diversity_over_time = []

    for gen in range(max_gens):
        pop.run(evaluate_population, 1)
        
        best_genome = max(pop.population.values(), key=lambda genome: genome.fitness or -float("inf"))
        
        if best_genome.fitness and best_genome.fitness > best_genome_fitness:
            best_genome_fitness = best_genome.fitness
            no_improvement = 0
        else:
            no_improvement += 1
        
        # Calculate genetic diversity at each generation
        diversity = compute_genetic_diversity(list(pop.population.items()))
        diversity_over_time.append(diversity)
        
        # Stop if target fitness is achieved or no improvement
        if best_genome_fitness >= target_fitness or no_improvement >= stop_after:
            avg_diversity = statistics.mean(diversity_over_time) if diversity_over_time else 0
            return best_genome_fitness, gen + 1, avg_diversity
    
    avg_diversity = statistics.mean(diversity_over_time) if diversity_over_time else 0
    return best_genome_fitness, max_gens, avg_diversity

# Run simulation with increasing population size
runs = 100
for run_id in range(1, runs + 1):
    print(f"Run {run_id}/{runs}")
    cfg = get_config()
    cfg.pop_size = 10 * run_id
    
    population_sizes.append(cfg.pop_size)
    
    # Measure elapsed time
    elapsed_time = timeit.timeit(lambda: evolve_population(cfg, target_fitness=0.95), number=1)
    time_results.append(elapsed_time)
    
    # Evolve population and gather results
    _, gens_required, avg_diversity = evolve_population(cfg, target_fitness=0.95)
    gen_counts.append(gens_required)
    diversity_results.append(avg_diversity)

# Print out stats
print("Time results:", time_results)
print("Average time:", statistics.mean(time_results))
print("Time standard deviation:", statistics.stdev(time_results))
print("Average generations:", statistics.mean(gen_counts))
print("Generations standard deviation:", statistics.stdev(gen_counts))
print("Average genetic diversity:", statistics.mean(diversity_results))

# Plot results using seaborn for better aesthetics
sns.set(style="whitegrid")  # Use seaborn for better style

# Set up figure and axis
plt.figure(figsize=(18, 6))

# Plot Time vs Population Size
plt.subplot(1, 3, 1)
sns.regplot(x=population_sizes, y=time_results, scatter_kws={"color": "royalblue"}, line_kws={"color": "orange"}, marker="o")
plt.xlabel("Population Size", fontsize=12)
plt.ylabel("Time to Reach Peak Fitness (seconds)", fontsize=12)
plt.title("Time vs Population Size", fontsize=14, fontweight="bold")
plt.grid(True)

# Plot Generations vs Population Size
plt.subplot(1, 3, 2)
sns.regplot(x=population_sizes, y=gen_counts, scatter_kws={"color": "seagreen"}, line_kws={"color": "orange"}, marker="o")
plt.xlabel("Population Size", fontsize=12)
plt.ylabel("Generations to Reach Peak Fitness", fontsize=12)
plt.title("Generations vs Population Size", fontsize=14, fontweight="bold")
plt.grid(True)

# Plot Genetic Diversity vs Population Size
plt.subplot(1, 3, 3)
sns.regplot(x=population_sizes, y=diversity_results, scatter_kws={"color": "crimson"}, line_kws={"color": "orange"}, marker="o")
plt.xlabel("Population Size", fontsize=12)
plt.ylabel("Average Genetic Diversity (Variance in Fitness)", fontsize=12)
plt.title("Genetic Diversity vs Population Size", fontsize=14, fontweight="bold")
plt.grid(True)

# Tighten layout and show/save
plt.tight_layout()
plt.savefig('neat_evolution_analysis_with_diversity.png')
plt.savefig('neat_evolution_analysis_with_diversity.pdf')
plt.show()
