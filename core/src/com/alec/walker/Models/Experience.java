package com.alec.walker.Models;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Experience {

	public INDArray state;
	public INDArray previousState;
	public INDArray target;
	public int action;
	public float reward;
	public float expectedReward;
	public INDArray expectedRewards;
	
	public Experience(INDArray state, INDArray previousState, INDArray expectedRewards, int action, float reward, float expectedReward) {
		super();
		
		this.action = action;
		this.state = state;
		this.previousState = previousState;
		this.expectedRewards = expectedRewards;
		this.reward = reward;
		this.expectedReward = expectedReward;
		
		target = Nd4j.create(expectedRewards.data());
		target.putScalar(action, reward);
	}
	
	
}
