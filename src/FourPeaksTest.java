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
import shared.Instance;

/**
 * A test using the 4 Peaks evaluation function
 * @author Tiffany Xu
 * @version 2.0
 */
public class FourPeaksTest {

    public static void main(String[] args) {
        int N = 10;
        int T = N / 10;
        int iterations = 10;
        while (N <= 100) {
            System.out.println("Current N: " + N);
            int[] ranges = new int[N];
            Arrays.fill(ranges, 2);
            EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
            Distribution odd = new DiscreteUniformDistribution(ranges);
            NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
            MutationFunction mf = new DiscreteChangeOneMutation(ranges);
            CrossoverFunction cf = new SingleCrossOver();
            Distribution df = new DiscreteDependencyTree(.1, ranges);
            HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
            GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
            ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);


            double curr = 0;
            System.out.println("Randomized Hill Climbing\n---------------------------------");
            long t = System.nanoTime();
            for (int i = 0; i < iterations; i++) {
                RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);

                FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
                fit.train();
                // System.out.println(ef.value(rhc.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
                if (ef.value(rhc.getOptimal()) == 2 * N - T - 1) {
                    curr++;
                }
            }
            System.out.println("Average: " + curr / 10 + "\nAverage Time: " + (((double) (System.nanoTime() - t)) / 1e10d) + "\n---------------------------------");

            curr = 0;
            t = System.nanoTime();
            System.out.println("Simulated Annealing\n---------------------------------");
            for (int i = 0; i < iterations; i++) {
                SimulatedAnnealing sa = new SimulatedAnnealing(1000, .95, hcp);
                FixedIterationTrainer fit = new FixedIterationTrainer(sa, 200000);
                fit.train();
                // System.out.println(ef.value(sa.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
                if (ef.value(sa.getOptimal()) == 2 * N - T - 1) {
                    curr++;
                }
            }
            System.out.println("Average: " + curr / 10 + "\nAverage Time: " + (((double) (System.nanoTime() - t)) / 1e10d) + "\n---------------------------------");

            curr = 0;
            t = System.nanoTime();
            System.out.println("Genetic Algorithm\n---------------------------------");
            for (int i = 0; i < iterations; i++) {
                StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(400, 100, 50, gap);
                FixedIterationTrainer fit = new FixedIterationTrainer(ga, 1000);
                fit.train();
                //System.out.println(ef.value(ga.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
                if (ef.value(ga.getOptimal()) == 2 * N - T - 1) {
                    curr++;
                }
            }
            System.out.println("Average: " + curr / 10 + "\nAverage Time: " + (((double) (System.nanoTime() - t)) / 1e10d) + "\n---------------------------------");

            curr = 0;
            t = System.nanoTime();
            System.out.println("MIMIC\n---------------------------------");
            for (int i = 0; i < iterations; i++) {
                MIMIC mimic = new MIMIC(200, 5, pop);
                FixedIterationTrainer fit = new FixedIterationTrainer(mimic, 1000);
                fit.train();
                //System.out.println(ef.value(mimic.getOptimal()) + ", " + (((double)(System.nanoTime() - t))/ 1e9d));
                if (ef.value(mimic.getOptimal()) == 2 * N - T - 1) {
                    curr++;
                }
            }
            System.out.println("Average: " + curr / 10 + "\nAverage Time: " + (((double) (System.nanoTime() - t)) / 1e10d) + "\n---------------------------------");
            N += 10;
        }
    }
}
