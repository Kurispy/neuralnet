import java.awt.FlowLayout;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileInputStream;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

public class Main {
    public static void main(String[] args) {
        NeuralNetwork neuralNetwork = new NeuralNetwork();
        final File maleFolder = new File("Male");
        final File femaleFolder = new File("Female");
        final File testFolder = new File("Test");
        final ArrayList<File> files = new ArrayList<>();
        final ArrayList<Boolean> genders = new ArrayList<>();
        
        if(args[0].equalsIgnoreCase("-train")) {
            for(File file : femaleFolder.listFiles()) {
                files.add(file);
                genders.add(Boolean.FALSE);
            }

            for(File file : maleFolder.listFiles()) {
                files.add(file);
                genders.add(Boolean.TRUE);
            }
            
            long seed = 3;
            Collections.shuffle(files, new Random(seed));
            Collections.shuffle(genders, new Random(seed));
        
            neuralNetwork.trainNetwork(files, genders);
            
            try(FileOutputStream fout = new FileOutputStream("weights")) {
                neuralNetwork.saveWeights(fout);
            }
            catch(Exception e) {
                System.err.println(e.getMessage());
            }
            
        }
        else if(args[0].equalsIgnoreCase("-validate")) {
            for(File file : femaleFolder.listFiles()) {
                files.add(file);
                genders.add(Boolean.FALSE);
            }

            for(File file : maleFolder.listFiles()) {
                files.add(file);
                genders.add(Boolean.TRUE);
            }
        
            neuralNetwork.validate(files, genders);
        }
        else if(args[0].equalsIgnoreCase("-test")) {
            files.addAll(java.util.Arrays.asList(testFolder.listFiles()));
            
            try(FileInputStream fin = new FileInputStream("weights")) {
                neuralNetwork.loadWeights(fin);
            }
            catch(Exception e) {
                System.err.println(e.getMessage());
            }
            
            neuralNetwork.testNetwork(files);
        }
        else if (args[0].equalsIgnoreCase("-visualize")){
        	
        	//reuse fin to get weights 
            try(FileInputStream fin = new FileInputStream("weights")) {
                painter(fin);
                
            }
            catch(Exception e) {
                System.err.println(e.getMessage());
            }

        }
        else
           System.err.println("Invalid argument.");
    }
    
    public static void painter(FileInputStream fin) {
	double[] store = new double[120 * 128];
    	byte[] buffer = new byte[(15360 + 1) * 8];
        ByteBuffer b = ByteBuffer.wrap(buffer);
        DoubleBuffer db = b.asDoubleBuffer();
        
        try {
            while(fin.available() > 0) {
                fin.read(buffer, 0, (15360 + 1) * 8);
            
                
                db.get();
                db.get(store);
                db.rewind();
                paint(store);
            }
        }
        catch (java.io.IOException e) {
            System.err.println(e.getMessage());
        }
        
        
	}
    
    
	public static void paint(double[] dataAsDouble) {
		
		// normalize all numbers
		double normhigh = 255, normlow = 0;
		double datamax = 0, datamin = 255;
		for (int k = 0; k < dataAsDouble.length; k++) {
			if (dataAsDouble[k] > datamax) {
		        datamax = dataAsDouble[k];
		    }
			if (dataAsDouble[k] < datamin) {
		        datamin = dataAsDouble[k];
		    }
		}
		
		int data[] = new int[120 * 128];
		for (int i = 0; i < dataAsDouble.length; i++){
			data[i] = (int) denormalize(dataAsDouble[i], datamin, datamax, normlow, normhigh);
		}
		
		
		//buffer image
	    BufferedImage image = new BufferedImage(128, 120, BufferedImage.TYPE_BYTE_GRAY);
	    image.getRaster().setPixels(0, 0, 128, 120, data);
	    
	    // display everything with a JFrame and a Jlabel using an icon
	    JFrame frame = new JFrame();
	    frame.getContentPane().setLayout(new FlowLayout());
	    frame.getContentPane().add(new JLabel(new ImageIcon(image)));
	    frame.pack();
	    frame.setVisible(true);
	}
	
	
	// modified from http://www.heatonresearch.com/wiki/Range_Normalization
	// normalizes a number to based on range
	// @param number, highs and lows 
	public static double denormalize(double x,double dataLow,double dataHigh,double normalizedLow, double normalizedHigh) {
		return ((x - dataLow) 
				/ (dataHigh - dataLow))
				* (normalizedHigh - normalizedLow) + normalizedLow;
	}
}
