package com.alec.walker.Models;

import java.util.Queue;

import org.joda.time.Duration;

import com.alec.walker.GamePreferences;
import com.badlogic.gdx.scenes.scene2d.ui.Table;
import com.google.common.collect.EvictingQueue;

public class BasicAgent extends BasicPlayer implements AbstractAgent {

	protected boolean			isDebug	= false;
	protected Duration			age;
	protected float				updateTime;
	protected float				updateTimer;

	protected float				learningRate;
	protected float				minLearningRate;
	protected float				maxLearningRate;
	protected float				learningRateDecay;
	protected float				mutationRate;
	protected float				randomness;
	protected float				minRandomness;
	protected float				maxRandomness;
	protected boolean			isManualControl;
	protected int				actionCount;
	protected float				futureDiscount;
	protected int				goal;
	// List of goals
	protected String[]			goals;
	protected int				previousAction;
	protected float				previousVelocity;
	protected float				previousValue;
	protected int				previousMaxAction;
	protected float				previousMaxActionValue;
	protected float				previousQValue;
	protected float				bestValue;
	protected float				worstValue;
	protected float				valueDelta;
	protected float				valueError;
	protected float				valueVelocity;
	protected float				impatience;
	protected float				explorationBonus;

	protected int				memoryCount;

	protected Queue<Integer>	previousActions;
	protected Queue<Integer>		previousValues;
	protected Queue<int[]>		previousStates;

	public BasicAgent() {

	}

	public void init(boolean withRandomness) {

		isManualControl = false;

		randomness = GamePreferences.instance.randomness;
		learningRate = GamePreferences.instance.learningRate;
		minLearningRate = GamePreferences.instance.minLearningRate;
		maxLearningRate = GamePreferences.instance.maxLearningRate;
		minRandomness = GamePreferences.instance.minRandomness;
		maxRandomness = GamePreferences.instance.maxRandomness;
		learningRateDecay = 0.9999f;
		futureDiscount = GamePreferences.instance.futureDiscount;
		explorationBonus = GamePreferences.instance.explorationBonus;

		memoryCount = 20;
		previousActions = EvictingQueue.create(memoryCount);
		previousStates = EvictingQueue.create(memoryCount);
		previousValues = EvictingQueue.create(memoryCount);

		updateTime = 0.0f;
		updateTimer = GamePreferences.instance.updateTimer;
		impatience = GamePreferences.instance.impatience;
		mutationRate  = GamePreferences.instance.mutationRate;

		if (withRandomness) {
			learningRate = Math.min(0.8f, (float) Math.random());
			learningRateDecay = learningRateDecay + (float) (0.00001 * Math.random());
			futureDiscount = 0.5f + (0.5f * (float) Math.random());
//			explorationBonus = 1 + 10 * (float) Math.random();
			updateTimer = (float) (updateTimer * Math.random());
			impatience = (float) (impatience * Math.random());
			
		}

	}

	public void setUpdateTimer(float updateTimer) {
		this.updateTimer = updateTimer;
	}

	public void setUpdateTimeout(float updateTimeout) {
		this.updateTimer = updateTimeout;
	}

	public float getUpdateTimer() {
		return updateTime;
	}

	public float getUpdateTimeout() {
		return updateTimer;
	}

	public String[] getGoals() {
		return goals;
	}

	public void setGoals(String[] goals) {
		this.goals = goals;
	}

	public float getLearningRate() {
		return learningRate;
	}

	public void setLearningRate(float learningRate) {
		this.learningRate = learningRate;
	}

	public float getMinLearningRate() {
		return minLearningRate;
	}

	public void setMinLearningRate(float minLearningRate) {
		this.minLearningRate = minLearningRate;
	}

	public float getMaxLearningRate() {
		return maxLearningRate;
	}

	public void setMaxLearningRate(float maxLearningRate) {
		this.maxLearningRate = maxLearningRate;
	}

	public float getLearningRateDecay() {
		return learningRateDecay;
	}

	public void setLearningRateDecay(float learningRateDecay) {
		this.learningRateDecay = learningRateDecay;
	}

	public float getRandomness() {
		return randomness;
	}

	public void setRandomness(float randomness) {
		this.randomness = randomness;
	}

	public float getMinRandomness() {
		return minRandomness;
	}

	public void setMinRandomness(float minRandomness) {
		this.minRandomness = minRandomness;
	}

	public float getMaxRandomness() {
		return maxRandomness;
	}

	public void setMaxRandomness(float maxRandomness) {
		this.maxRandomness = maxRandomness;
	}

	public boolean getIsManualControl() {
		return isManualControl;
	}

	public void setIsManualControl(boolean isManualControl) {
		this.isManualControl = isManualControl;
	}

	@Override
	public void setManualControl(boolean isManuControl) {
		this.isManualControl = isManuControl;
	}

	@Override
	public boolean getManualControl() {
		return isManualControl;
	}

	public void setValueDelta(float valueDelta) {
		this.valueDelta = valueDelta;
	}

	public float getValueError() {
		return valueError;
	}

	public void cycleGoal() {
		goal = (goal + 1) % goals.length;
	}

	public int getGoal() {
		return goal;
	}

	public void setGoal(int goal) {
		this.goal = goal;
	}

	public float fastSig(float x) {
		if (x < -10) {
			return 0;
		} else if (x > 10) {
			return 1;
		} else {
			return (float) (1 / (1 + Math.exp(-x)));
		}
	}

	@Override
	public Table getLearningMenu() {
		// TODO Auto-generated method stub
		return null;
	}

	public float getFutureDiscount() {
		return futureDiscount;
	}

	public void setFutureDiscount(float futureDiscount) {
		this.futureDiscount = futureDiscount;
	}

	public float getExplorationBonus() {
		return explorationBonus;
	}

	public void setExplorationBonus(float explorationBonus) {
		this.explorationBonus = explorationBonus;
	}

	public boolean getIsDebug() {
		return isDebug;
	}

	public void setIsDebug(boolean isDebug) {
		this.isDebug = isDebug;
	}

	public void setImpatience(float value) {
		impatience = value;
		
	}

	public int getMemoryCount() {
		return memoryCount;
	}

	public void setMemoryCount(int memoryCount) {
		this.memoryCount = memoryCount;
	}

	public float getImpatience() {
		return impatience;
	}

	public float getMutationRate() {
		return mutationRate;
	}

	public void setMutationRate(float mutationRate) {
		this.mutationRate = mutationRate;
	}
	
	

}
