package com.alec.walker.Models;

import java.util.ArrayList;

import com.badlogic.gdx.InputAdapter;
import com.badlogic.gdx.graphics.Camera;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.physics.box2d.Body;
import com.badlogic.gdx.scenes.scene2d.ui.Table;

public class BasicPlayer extends InputAdapter implements Player {

	protected ArrayList<Body>	bodies;
	protected Body				body;
	public String name;

	

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


	



}