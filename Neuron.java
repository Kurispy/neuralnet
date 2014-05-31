import java.util.ArrayList;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;

public abstract class Neuron {
    protected int numInputs;
    protected double[] inputWeights;
    protected ArrayList<Double> inputs;
    protected double bias;
    protected double output;
    protected double gradient;
    
    Neuron(final int numInputs, ArrayList<Double> inputs, double initalWeightClamp) {
        this.numInputs = numInputs;
        this.inputs = inputs;
        bias = 2 * (Math.random() - 0.5) * initalWeightClamp;
        inputWeights = new double[numInputs];
        for(int i = 0; i < numInputs; i++) {
            inputWeights[i] = 2 * (Math.random() - 0.5) * initalWeightClamp;
        }
    }
    
    public void computeOutput() {
        double newOutput = bias;
        for(int i = 0; i < numInputs; i++) {
            newOutput += inputWeights[i] * inputs.get(i);
        }
        output = activationFunction(newOutput);
    }
    
    public double getOutput() {
        return output;
    }
    
    public double getGradient() {
        return gradient;
    }
    
    public void updateWeights(final double learningFactor) {
        bias += learningFactor * gradient;
        for(int i = 0; i < numInputs; i++) {
            inputWeights[i] += learningFactor * gradient * inputs.get(i);
        }
    }
    
    public double getWeight(int index) {
        return inputWeights[index];
    }
    
    protected void saveWeights(FileOutputStream fout) {
        ByteBuffer b = ByteBuffer.allocate((numInputs + 1) * 8);
        DoubleBuffer db = b.asDoubleBuffer();
        db.put(bias);
        db.put(inputWeights);
        
        try {
            fout.write(b.array());
        }
        catch (java.io.IOException e) {
            System.err.println(e.getMessage());
        }
    }
    
    protected void loadWeights(FileInputStream fin) {
        byte[] buffer = new byte[(numInputs + 1) * 8];
        
        try {
            fin.read(buffer, 0, (numInputs + 1) * 8);
        }
        catch (java.io.IOException e) {
            System.err.println(e.getMessage());
        }
        
        ByteBuffer b = ByteBuffer.wrap(buffer);
        DoubleBuffer db = b.asDoubleBuffer();
        bias = db.get();
        db.get(inputWeights);
    }
    
    protected double activationFunction(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }
}
