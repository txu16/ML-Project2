import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test using the flip flop evaluation function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FlipFlopTest {
    /** The n value */
    private static final int N = 80;

    public static void main(String[] args) {
        int N = 10;
        while (N <= 100) {
            System.out.println("Current N: " + N);
            int[] ranges = new int[N];
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new FlipFlopEvaluationFunction();
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new SingleCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);


            System.out.println("Randomized Hill Climbing\n---------------------------------");
            long t = System.nanoTime();
            RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
            FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
            fit.train();
            System.out.println("Average: " + ef.value(rhc.getOptimal())+ "\nAverage Time: " + (((double) (System.nanoTime() - t)) / 1e10d));

            System.out.println("Simulated Annealing \n---------------------------------");
            t = System.nanoTime();
            SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
            fit = new FixedIterationTrainer(sa, 200000);
            fit.train();
            System.out.println("Average: " + ef.value(sa.getOptimal())+ "\nAverage Time: " + (((double) (System.nanoTime() - t)) / 1e10d));

            System.out.println("Genetic Algorithm\n---------------------------------");
            t = System.nanoTime();
            StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 20, gap);
            fit = new FixedIterationTrainer(ga, 1000);
            fit.train();
            System.out.println("Average: " + ef.value(ga.getOptimal())+ "\nAverage Time: " + (((double) (System.nanoTime() - t)) / 1e10d));

            System.out.println("MIMIC \n---------------------------------");
            t = System.nanoTime();
            MIMIC mimic = new MIMIC(200, 5, pop);
            fit = new FixedIterationTrainer(mimic, 1000);
            fit.train();
            System.out.println("Average: " + ef.value(mimic.getOptimal())+ "\nAverage Time: " + (((double) (System.nanoTime() - t)) / 1e10d));
            N += 10;
        }
    }
}
