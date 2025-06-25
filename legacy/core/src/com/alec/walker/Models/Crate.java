package com.alec.walker.Models;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.Sprite;
import com.badlogic.gdx.physics.box2d.Body;
import com.badlogic.gdx.physics.box2d.BodyDef;
import com.badlogic.gdx.physics.box2d.BodyDef.BodyType;
import com.badlogic.gdx.physics.box2d.FixtureDef;
import com.badlogic.gdx.physics.box2d.PolygonShape;
import com.badlogic.gdx.physics.box2d.World;

public class Crate {

	Body body;
	float width, height;
	boolean isContacting = false;
	boolean isDestroyed = false;
	
	public Crate(World world, float width, float height, float x , float y, short categoryBits, int maskBits) {
		body = null;
		this.width = width;
		this.height = height;
		
		// body def
		BodyDef bodyDef = new BodyDef();
		bodyDef.type = BodyType.DynamicBody;
		bodyDef.position.set(x,y);
		body = world.createBody(bodyDef);
		
		// shape
		PolygonShape shape = new PolygonShape();
		shape.setAsBox(width, height);
		
		// fixture
		FixtureDef fixtureDef = new FixtureDef();
		fixtureDef.shape = shape;
		fixtureDef.friction = .33f;
		fixtureDef.density = .65f;
		fixtureDef.restitution = .2f;
		fixtureDef.filter.categoryBits = categoryBits;
		fixtureDef.filter.maskBits = (short)maskBits;
		
		body.createFixture(fixtureDef);
		
		// sprite
		Sprite crateSprite = new Sprite(new Texture("data/img/crate.png"));
		crateSprite.setSize(width * 2, height * 2);
		crateSprite.setOrigin(crateSprite.getWidth() / 2, crateSprite.getHeight() / 2);
		body.setUserData(crateSprite);
		
	}
	
	public void render() {
		
	}
	
	public void startContact() {
		isContacting = true;
	}
	
	public void endContact() {
		isContacting = false;
	}
	
	public void destroy() {
		isDestroyed = true;
	}
	
	public Body getBody() {
		return body;
	}
	
	

}
