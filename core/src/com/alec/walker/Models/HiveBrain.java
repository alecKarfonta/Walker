package com.alec.walker.Models;

import java.util.ArrayList;
import java.util.Random;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class HiveBrain {

	public MultiLayerNetwork		criticBrain;
	public int						memoryCount		= 10;
	public int						actionCount		= 4;
	public int						weightCount		= 10;

	public static final int			seed			= 12345;
	public static final int			iterations		= 1;
	public static final double		learningRateNN	= 0.0001f;
	public static final Random		rng				= new Random(seed);
	public ArrayList<Experience>	experiences;

	public void init(int memoryCount, int actionCount, int weightCount) {

		this.memoryCount = memoryCount;
		this.actionCount = actionCount;
		this.weightCount = weightCount;

		// int numInput = weightCount;
		int numInput = weightCount;
		int numOutputs = actionCount;
		// int numOutputs = weightCount * actionCount + 1;
		int nHidden = 128;
		// Create the network
		criticBrain = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.learningRate(learningRateNN)
				.weightInit(WeightInit.RELU)
				.updater(Updater.ADAM)
				.list()
				.layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
						.activation("relu")
						.dropOut(0.1)
						.build())

				.layer(1, new DenseLayer.Builder().nIn(nHidden).nOut(nHidden)
						.activation("relu")
						.dropOut(0.2)
						.build())
				.layer(2, new DenseLayer.Builder().nIn(nHidden * 2).nOut(nHidden)
						.activation("relu")
						.dropOut(0.1)
						.build())
				.layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
						.activation("linear")
						.nIn(nHidden).nOut(numOutputs).build())
				.pretrain(false).backprop(true).build()
				);

		criticBrain.init();

	}

}
