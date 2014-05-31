import java.util.ArrayList;

public class HiddenNeuron extends Neuron {
    
    HiddenNeuron(final int numInputs, ArrayList<Double> inputs, double initalWeightClamp) {
        super(numInputs, inputs, initalWeightClamp);
    }
    
    public void computeGradient(final double targetOutput, double weight, double outputGradient) {
        gradient = output * (1 - output) * weight * outputGradient;
    }
}
