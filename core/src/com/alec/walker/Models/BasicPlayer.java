package com.alec.walker.Models;

import java.util.ArrayList;

import com.badlogic.gdx.InputAdapter;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.physics.box2d.Body;
import com.badlogic.gdx.physics.box2d.Joint;

public class BasicPlayer extends InputAdapter implements Player {

	public Body				body;
	public String name;
	public int	isTouchingGround;
	
	public float mutationRate;

	

	public void sendHome() {
		System.out.println("sendHome()");
		for (Body body : getBodies()) {
			body.setTransform(new Vector2(0, 15), (float) Math.toRadians(0));
			body.setLinearVelocity(new Vector2(0, 0));
		}

	}



	public void learnFrom(BasicPlayer basicPlayer, float transferRate) {
		
	}

	public void learnFromAll(ArrayList<BasicPlayer> allPlayers, float learningRate) {
		
	}

	public void setMutationRate(float value) {
		System.out.println("BasicPlayer.setMutationRate()");
	}


	@Override
	public Body getBody() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public ArrayList<Body> getBodies() {
		// TODO Auto-generated method stub
		return null;
	}


	@Override
	public ArrayList<String> getStats() {
		// TODO Auto-generated method stub
		return null;
	}


	



}