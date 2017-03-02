package com.alec.walker.Controllers;

import com.badlogic.gdx.physics.box2d.*;
import com.badlogic.gdx.physics.box2d.BodyDef.BodyType;

public class BallFactory {
	public static final BallFactory instance = new BallFactory();
	
	public void createBall(World world, float x, float y, float r, short categoryBits, short maskBits) {
		// body def
		BodyDef bodyDef = new BodyDef();
		bodyDef.type = BodyType.DynamicBody;
		bodyDef.position.set(x, y);
		bodyDef.linearDamping = 0;
		// shape
		CircleShape shape = new CircleShape();
		shape.setRadius(r);

		// fixture
		FixtureDef fixtureDef = new FixtureDef();
		fixtureDef.shape = shape;
		fixtureDef.friction = .33f;
		fixtureDef.density = .25f;
		fixtureDef.restitution = .9f;
		fixtureDef.filter.categoryBits = categoryBits;
		fixtureDef.filter.maskBits = (short) maskBits;

		Body body = world.createBody(bodyDef);
		body.createFixture(fixtureDef);
		// sprite
//		Sprite crateSprite = new Sprite(new Texture("data/img/crate.png"));
//		crateSprite.setSize(width * 2, height * 2);
//		crateSprite.setOrigin(crateSprite.getWidth() / 2, crateSprite.getHeight() / 2);
//		body.setUserData(crateSprite);
	}
}
