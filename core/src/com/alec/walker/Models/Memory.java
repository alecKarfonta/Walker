package com.alec.walker.Models;

import org.nd4j.linalg.api.ndarray.INDArray;

import com.google.common.collect.EvictingQueue;

public class Memory {

	public int						memoryCount			= 10;
	
	public EvictingQueue<INDArray>	currentActorTarget;
	public EvictingQueue<INDArray>	currentStates;
	public EvictingQueue<INDArray>	previousStates;
	public EvictingQueue<Float>	currentRewards;
	public EvictingQueue<Integer>	previousActions;
	
}
