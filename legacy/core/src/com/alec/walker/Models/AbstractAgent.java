package com.alec.walker.Models;

import java.util.ArrayList;

import com.badlogic.gdx.graphics.Camera;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.physics.box2d.Body;
import com.badlogic.gdx.scenes.scene2d.ui.Table;

public interface AbstractAgent {

	public Table getLearningMenu();

	public float getUpdateTimer();

	public void setUpdateTimer(float updateTimer);

	public void setManualControl(boolean isManuControl);

	public boolean getManualControl();

	public void setLearningRate(float learningRate);

	public float getLearningRate();

	public float getMinLearningRate();

	public float getMaxLearningRate();

	public void setMinLearningRate(float minLearningRate);

	public void setMaxLearningRate(float maxLearningRate);

	public float getRandomness();

	public void setRandomness(float randomness);

	public float getMinRandomness();

	public float getMaxRandomness();

	public void setMinRandomness(float minRandomness);

	public void setMaxRandomness(float maxRandomness);

}
