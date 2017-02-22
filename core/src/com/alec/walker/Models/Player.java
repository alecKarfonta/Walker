package com.alec.walker.Models;

import java.util.ArrayList;

import com.badlogic.gdx.InputAdapter;
import com.badlogic.gdx.graphics.Camera;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.physics.box2d.Body;
import com.badlogic.gdx.scenes.scene2d.ui.Table;

public interface Player {

	public Body getBody();

	public ArrayList<Body> getBodies();

	public ArrayList<String> getStats();

	public void sendHome();


	

}
