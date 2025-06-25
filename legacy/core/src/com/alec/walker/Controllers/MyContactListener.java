package com.alec.walker.Controllers;

import com.alec.walker.Views.Play;
import com.badlogic.gdx.physics.box2d.Contact;
import com.badlogic.gdx.physics.box2d.ContactImpulse;
import com.badlogic.gdx.physics.box2d.ContactListener;
import com.badlogic.gdx.physics.box2d.Manifold;

public class MyContactListener implements ContactListener {
	private static final String TAG = MyContactListener.class.getName();
	
	private WorldController worldController; // reference play so you can call functions
	
	public MyContactListener(WorldController worldController) {
		this.worldController = worldController;
	}
	
	@Override
	public void beginContact(Contact contact) {
		
	}

	@Override
	public void endContact(Contact contact) {
	}

	@Override
	public void preSolve(Contact contact, Manifold oldManifold) {
	
		
	}
	@Override
	public void postSolve(Contact contact, ContactImpulse impulse) {
		if (impulse.getNormalImpulses()[0] > 200) {
			
		}
	}

}
