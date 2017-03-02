package com.alec.walker.Controllers;

import com.alec.walker.Models.StandingCrate;
import com.badlogic.gdx.physics.box2d.Contact;
import com.badlogic.gdx.physics.box2d.ContactImpulse;
import com.badlogic.gdx.physics.box2d.ContactListener;
import com.badlogic.gdx.physics.box2d.Manifold;

public class MyContactListener implements ContactListener {
	private static final String	TAG	= MyContactListener.class.getName();

	private WorldController		worldController;							// reference play so you can call functions

	public MyContactListener(WorldController worldController) {
		this.worldController = worldController;
	}

	@Override
	public void beginContact(Contact contact) {
		if (contact.getFixtureA().getBody().getUserData() != null) {
			if (contact.getFixtureA().getBody().getUserData() instanceof StandingCrate) {
				if (contact.getFixtureB().getBody().getUserData() instanceof String) {
//					System.out.println("Contact (): StandingCrate ->  " + contact.getFixtureB().getUserData());
				}
				((StandingCrate)contact.getFixtureA().getBody().getUserData()).isTouchingGround = 1;
			} else if (contact.getFixtureB().getBody().getUserData() instanceof StandingCrate) {
				if (contact.getFixtureA().getBody().getUserData() instanceof String) {
//					System.out.println("Contact (): StandingCrate ->  " + contact.getFixtureA().getBody().getUserData());
				}
				((StandingCrate)contact.getFixtureB().getBody().getUserData()).isTouchingGround = 1;
			}
			
		}
	}

	@Override
	public void endContact(Contact contact) {if (contact.getFixtureA().getBody().getUserData() != null) {
		if (contact.getFixtureA().getBody().getUserData() instanceof StandingCrate) {
			if (contact.getFixtureB().getBody().getUserData() instanceof String) {
//				System.out.println("Contact (): StandingCrate ->  " + contact.getFixtureB().getUserData());
			}
			((StandingCrate)contact.getFixtureA().getBody().getUserData()).isTouchingGround = 0;
		} else if (contact.getFixtureB().getBody().getUserData() instanceof StandingCrate) {
			if (contact.getFixtureA().getBody().getUserData() instanceof String) {
//				System.out.println("Contact (): StandingCrate ->  " + contact.getFixtureA().getBody().getUserData());
			}
			((StandingCrate)contact.getFixtureB().getBody().getUserData()).isTouchingGround = 0;
		}
		
	}
	}

	@Override
	public void preSolve(Contact contact, Manifold oldManifold) {

		// + " -> " + contact.getFixtureB().getUserData()
		// + " ( " + impulse.getNormalImpulses()[0] + " N)");
		// if (impulse.getNormalImpulses()[0] > 200) {
		//
		// }


	}

	@Override
	public void postSolve(Contact contact, ContactImpulse impulse) {

	}

}
