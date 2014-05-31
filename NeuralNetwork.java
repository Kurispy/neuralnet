import java.io.*;
import java.util.*;

public class NeuralNetwork {
    private static final double CLASSIFICATION_TARGET_MALE = 0.9;
    private static final double CLASSIFICATION_TARGET_FEMALE = 0.1;
    private static final int NUM_INPUT_NODES = 15360;
    private static final int NUM_HIDDEN_NODES = 40;
    private static final double LEARNING_FACTOR = 0.3;
    private static final double INTIAL_WEIGHT_VALUE_CLAMP = 0.5;
    
    private final ArrayList<Double> inputLayer = new ArrayList<>();
    private final ArrayList<HiddenNeuron> hiddenLayer = new ArrayList<>();
    private final ArrayList<Double> hiddenLayerOutputs = new ArrayList<>();
    private final OutputNeuron outputLayer = new OutputNeuron(NUM_HIDDEN_NODES, hiddenLayerOutputs, INTIAL_WEIGHT_VALUE_CLAMP);
    {
        for(int i = 0; i < NUM_HIDDEN_NODES; i++) {
            hiddenLayer.add(new HiddenNeuron(NUM_INPUT_NODES, inputLayer, INTIAL_WEIGHT_VALUE_CLAMP));
        }
    }
    
    public double trainNetwork(ArrayList<File> files, ArrayList<Boolean> genders) {
        int matches = 0;
        for(int i = 0; i < files.size(); i++) {
            readInputs(files.get(i));
            computeOutput();
            double certainty = classify();
            //System.out.println(genders.get(i) + " " + certainty);
            if(genders.get(i) == (certainty > 0))
                matches++;
            if(genders.get(i)) 
                updateWeights(CLASSIFICATION_TARGET_MALE);
            else
                updateWeights(CLASSIFICATION_TARGET_FEMALE);
        }
        
        double accuracy = ((double) matches) / ((double) files.size());
        return accuracy;
    }
    
    public void testNetwork(ArrayList<File> files) {
        for(File file : files) {
            readInputs(file);
            computeOutput();
            double certainty = classify();
            if(certainty > 0) {
                System.out.println(file.getName() + ": " + "MALE " + String.format("%.2f", certainty));
            }
            else {
                System.out.println(file.getName() + ": " + "FEMALE " + String.format("%.2f", -certainty));
            }
        }
    }
    
    public double testNetwork(ArrayList<File> files, ArrayList<Boolean> genders) {
        int matches = 0;
        for(int i = 0; i < files.size(); i++) {
            readInputs(files.get(i));
            computeOutput();
            if(genders.get(i) == (classify() > 0))
                matches++;
        }
        
        double accuracy = ((double) matches) / ((double) files.size());
        return accuracy;
    }
    
    public void validate(ArrayList<File> files, ArrayList<Boolean> genders) {
        int foldSize = files.size() / 5;
        ArrayList<File> trainingFiles, testFiles;
        ArrayList<Boolean> trainingGenders, testGenders;
        for(int i = 0; i < 10; i++) {
            double trainingMean = 0.0;
            double testMean = 0.0;
            double trainingM2 = 0.0;
            double testM2 = 0.0;
            
            long seed = System.nanoTime();
            Collections.shuffle(files, new Random(seed));
            Collections.shuffle(genders, new Random(seed));
            
            for(int j = 0; j < 5; j++) {
                testFiles = new ArrayList<>(files.subList(j * foldSize, j == 4 ? files.size() : (j + 1) * foldSize));
                testGenders = new ArrayList<>(genders.subList(j * foldSize, j == 4 ? genders.size() : (j + 1) * foldSize));
                trainingFiles = new ArrayList<>(files);
                trainingFiles.subList(j * foldSize, j == 4 ? files.size() : (j + 1) * foldSize).clear();
                trainingGenders = new ArrayList<>(genders);
                trainingGenders.subList(j * foldSize, j == 4 ? genders.size() : (j + 1) * foldSize).clear();
                
                double trainingAccuracy = trainNetwork(trainingFiles, trainingGenders);
                double trainingDelta = trainingAccuracy - trainingMean;
                trainingMean += trainingDelta / (j + 1);
                trainingM2 += trainingDelta * (trainingAccuracy - trainingMean);
                        
                double testAccuracy = testNetwork(testFiles, testGenders);
                double testDelta = testAccuracy - testMean;
                testMean += testDelta / (j + 1);
                testM2 += testDelta * (testAccuracy - testMean);
            }
            
            double trainingSD = Math.sqrt(trainingM2 / 4);
            double testSD = Math.sqrt(testM2 / 4);
            
            System.out.println("TEST " + (i + 1) + " RESULTS");
            System.out.println("Trainging Mean: " + trainingMean + " Training Standard Deviation: " + trainingSD);
            System.out.println("Test Mean: " + testMean + " Test Standard Deviation: " + testSD);
        }
    }
    
    public void saveWeights(FileOutputStream fout) {
        for(Neuron neuron : hiddenLayer) {
            neuron.saveWeights(fout);
        }
        outputLayer.saveWeights(fout);
    }
    
    public void loadWeights(FileInputStream fin) {
        for(Neuron neuron : hiddenLayer) {
            neuron.loadWeights(fin);
        }
        outputLayer.loadWeights(fin);
    }
    
    private void updateWeights(double targetOutput) {
        outputLayer.computeGradient(targetOutput);
        for(int i = 0; i < NUM_HIDDEN_NODES; i++) {
            hiddenLayer.get(i).computeGradient(targetOutput, outputLayer.getWeight(i), outputLayer.getGradient());
            hiddenLayer.get(i).updateWeights(LEARNING_FACTOR);
        }
        outputLayer.updateWeights(LEARNING_FACTOR);
    }
    
    private void computeOutput() {
        hiddenLayerOutputs.clear();
        for(Neuron neuron : hiddenLayer) {
            neuron.computeOutput();
            hiddenLayerOutputs.add(neuron.getOutput());
        }
        outputLayer.computeOutput();
    }
    
    private double getOutput() {
        return outputLayer.getOutput();
    }
    
    private void readInputs(File file) {
        inputLayer.clear();
        try (
                FileInputStream fstream = new FileInputStream(file);
                BufferedReader br = 
                        new BufferedReader(new InputStreamReader(fstream))
        ) {
            String strLine;
            while ((strLine = br.readLine()) != null){
                String[] line = strLine.split(" ");
                for (String value : line) {
                    inputLayer.add((double) Integer.parseInt(value));
                }
            }
        }
        catch (Exception e){
            System.err.println(e.getMessage());
        }
    }
    
    private double classify() {
        double output = getOutput() - 0.5;
        double normalM = CLASSIFICATION_TARGET_MALE - 0.5;
        double normalF = CLASSIFICATION_TARGET_FEMALE - 0.5;
        double certainty;
        if(output > 0) {
            certainty = 100 * (output / normalM);
            return certainty;
        }
        else if(output < 0) {
            certainty = 100 * (output / normalF);
            return -certainty;
        }
        else {
            certainty = 100 * (output / normalF);
            return -certainty;
        }
    }
}
