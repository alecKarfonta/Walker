package com.alec.walker.Models;

import com.badlogic.gdx.InputAdapter;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.physics.box2d.Body;
import com.badlogic.gdx.physics.box2d.Joint;

import java.util.ArrayList;

public class BasicPlayer extends InputAdapter implements Player {

	public ArrayList<Body>	bodies;
	public ArrayList<Joint>	joints;
	public Body				body;
	public String name;
	public int	isTouchingGround;

	

	public void sendHome() {
		System.out.println("sendHome()");
		for (Body body : getBodies()) {
			body.setTransform(new Vector2(0, 15), (float) Math.toRadians(0));
			body.setLinearVelocity(new Vector2(0, 0));
		}

	}

	@Override
	public Body getBody() {
		return body;
	}

	@Override
	public ArrayList<Body> getBodies() {
		return bodies;
	}

	@Override
	public ArrayList<String> getStats() {
		return null;
	}

	public ArrayList<Joint> getJoints() {
		return joints;
	}


	



}