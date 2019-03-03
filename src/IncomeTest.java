import opt.*;
import opt.example.*;
import shared.*;
import func.nn.backprop.*;
import shared.filt.TestTrainSplitFilter;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 *
 * @author Tiffany Xu
 * @version 2.0
 */
public class IncomeTest {
    private static Instance[] instances = initializeInstances();
    private static Instance[] trainInstances;
    private static Instance[] testInstances;

    private static int inputLayer = 14, hiddenLayer = 11, outputLayer = 1, trainingIterations = 100;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static ErrorMeasure measure = new SumOfSquaresError();
    static int trial = 1;

    static int tIter = 100;

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[1];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[1];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[1];
    private static String[] oaNames = {
            "SA T = 1E9, C = 0.65"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        TestTrainSplitFilter ttsf = new TestTrainSplitFilter(80);
        ttsf.filter(set);
        DataSet training = ttsf.getTrainingSet();
        DataSet testing = ttsf.getTestingSet();
        trainInstances = training.getInstances();
        testInstances = testing.getInstances();
        for (int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[]{inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }
        System.out.println("Note: training iteration for genetic algorithm will be 1/10 less");
        while (trainingIterations <= 2000) {
            System.out.println("Current Training Iteration " + trainingIterations + ":");

            oa[0] = new SimulatedAnnealing(1E9, .65, nnop[0]);

            results = "";



            for (int i = 0; i < oa.length; i++) {
                double totalTraining = 0;
                double totalTesting = 0;
                double totalCorrect = 0;
                double totalInCorrect = 0;
                for (int q = 0; q < trial; q++) {
                    double start = System.nanoTime();
                    double end, trainingTime, testingTime, correct = 0, incorrect = 0;
                    train(oa[i], networks[i], oaNames[i]); //trainer.train();
                    end = System.nanoTime();
                    trainingTime = end - start;
                    trainingTime /= Math.pow(10, 9);
                    Instance optimalInstance = oa[i].getOptimal();
                    networks[i].setWeights(optimalInstance.getData());
                    start = System.nanoTime();
                    for (int j = 0; j < instances.length; j++) {
                        networks[i].setInputValues(instances[j].getData());
                        networks[i].run();

                        double predicted = Double.parseDouble(instances[j].getLabel().toString());
                        double actual = Double.parseDouble(networks[i].getOutputValues().toString());
                        double trash = (Math.abs(predicted - actual) < 0.5) ? correct++ : incorrect++;

                    }
                    end = System.nanoTime();
                    testingTime = end - start;
                    testingTime /= Math.pow(10, 9);
                    totalTesting += testingTime;
                    totalTraining += trainingTime;
                    totalCorrect += correct;
                    totalInCorrect += incorrect;
                    //System.out.println("Trial " +(q + 1) + " for " + oaNames[i] + ": Percent correctly classified: "
                    //    + df.format(correct / (correct + incorrect) * 100) );
                }
                results += "\nAverage Results for " + oaNames[i] + ": \nCorrectly classified " + totalCorrect/trial + " instances." +
                        "\nIncorrectly classified " + totalInCorrect/trial + " instances.\nPercent correctly classified: "
                        + df.format(totalCorrect / (totalCorrect + totalInCorrect) * 100) + "%\nTraining time: " + df.format(totalTraining/trial)
                        + " seconds\nTesting time: " + df.format(totalTesting/trial) + " seconds\n";
            }
            System.out.println(results);
            trainingIterations += 100;
        }
    }

    private static int argmax(Instance vec) {
        double max = -100000.0;
        int best_ix = 0;
        for (int i = 0; i < vec.size(); i++) {
            double val = vec.getContinuous(i);
            if (val > max) {
                max = val;
                best_ix = i;
            }
        }
        return best_ix;
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");
        int realIter = tIter;
        if (oaName.charAt(0) == 'G') {
            realIter /= 10;
        }
        int totalTrainError = 0;
        int totalTestError = 0;
        for(int i = 0; i < realIter; i++) {
            oa.train();
            double error = 0;
            double testError = 0;
            for(int j = 0; j < trainInstances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }
            for (int j = 0; j < testInstances.length; j++) {
                network.setInputValues(testInstances[j].getData());
                network.run();

                Instance output = testInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                testError += measure.value(output, example);
            }
            System.out.println("training error: " + df.format(error) + ", test error: " + df.format(testError));

            //System.out.println(df.format(error));
            totalTrainError += error;
            totalTestError += testError;
        }
        //System.out.println(df.format(totalError/realIter));
        System.out.println("training error: " + df.format(totalTrainError/realIter) + ", test error: " + df.format(totalTestError/realIter));
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[16280][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("newAdult.txt")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[14]; // 16 attributes
                attributes[i][1] = new double[1];
                for(int j = 0; j < 14; j++) {
                    attributes[i][0][j] = Double.parseDouble(scan.next());
                }
                int num = (int)Double.parseDouble(scan.next());
                attributes[i][1][0] = num;

            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }


        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        return instances;
    }
}