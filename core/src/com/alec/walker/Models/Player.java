package com.alec.walker.Models;

import com.badlogic.gdx.physics.box2d.Body;

import java.util.ArrayList;

public interface Player {

	public Body getBody();

	public ArrayList<Body> getBodies();

	public ArrayList<String> getStats();

	public void sendHome();


	

}
