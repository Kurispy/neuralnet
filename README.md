neuralnet
=========

Gender classification of images using an artificial neural network.

Compile and run:
javac *.java
java Main

Options:
-train
Trains the network based on the files that are located in the “Male” and “Female” directories. When training is finished, the program saves all weights and biases to a file called “weights”.
-test
Tests the network based on the files that are located in the “Test” directory. For each file, the program prints a single line in this format: <filename>: {MALE, FEMALE} <confidence (in percent)>
-validate
Performs 5-fold cross-validation. Prints statistics for each of the 10 cross-validations. The program randomly shuffles the order of the list of files in between every cross-validation.
-visualize
Visualizes the weights of each hidden node, using the saved “weights” file. The program must be manually terminated to exit.
