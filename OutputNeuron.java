import java.util.ArrayList;

public class OutputNeuron extends Neuron {
    
    OutputNeuron(final int numInputs, ArrayList<Double> inputs, double initalWeightClamp) {
        super(numInputs, inputs, initalWeightClamp);
    }
    
    public void computeGradient(final double targetOutput) {
        gradient = output * (1 - output) * (targetOutput - output);
    }
}
