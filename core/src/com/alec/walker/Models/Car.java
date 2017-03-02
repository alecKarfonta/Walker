package com.alec.walker.Models;

import com.alec.walker.Constants;
import com.alec.walker.StringHelper;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input.Keys;
import com.badlogic.gdx.graphics.Camera;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.ParticleEmitter;
import com.badlogic.gdx.graphics.g2d.Sprite;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.physics.box2d.*;
import com.badlogic.gdx.physics.box2d.BodyDef.BodyType;
import com.badlogic.gdx.physics.box2d.joints.RevoluteJoint;
import com.badlogic.gdx.physics.box2d.joints.RevoluteJointDef;
import com.badlogic.gdx.physics.box2d.joints.WheelJoint;
import com.badlogic.gdx.physics.box2d.joints.WheelJointDef;
import com.badlogic.gdx.scenes.scene2d.ui.Table;

import java.util.ArrayList;

public class Car extends BasicAgent {
	
	private Body chassis, leftWheel, rightWheel, sled;
	private WheelJoint leftAxis, rightAxis;
	private RevoluteJoint sledJoint;
	private int motorSpeed = 15;
	private float motorTorque = 500.0f;
	private ParticleEmitter exhaust;
	private float x , y;
	private float width, height;
	
	private float speed = 0f;
	private float speedDecay = .1f;
	private boolean holdMotors = true;
	
	public Car(World world, 
			float x, float y, 
			float width, float height) {
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
		
		BodyDef bodyDef = new BodyDef();
		bodyDef.type = BodyType.DynamicBody;
		bodyDef.position.set(x,y);
		bodyDef.allowSleep = false;
		
		FixtureDef fixtureDef = new FixtureDef();
		FixtureDef wheelFixtureDef = new FixtureDef();
		FixtureDef sledFixtureDef = new FixtureDef();
		FixtureDef radarFixtureDef = new FixtureDef();

		fixtureDef.density = 2;
		fixtureDef.friction = .4f;
		fixtureDef.restitution = .2f;

		wheelFixtureDef.density = fixtureDef.density * 3f;
		wheelFixtureDef.friction = 10;
		wheelFixtureDef.restitution = 0.0001f;

		sledFixtureDef.density = .1f;
		sledFixtureDef.friction = 0;
		sledFixtureDef.restitution = 0;

		radarFixtureDef.density = 0;
		radarFixtureDef.friction = 0;
		radarFixtureDef.restitution = 0;
		radarFixtureDef.isSensor = true;
		radarFixtureDef.filter.categoryBits = Constants.FILTER_RADAR;
		radarFixtureDef.filter.maskBits = Constants.FILTER_CRATE;
		
		// create the chassis
		PolygonShape chassisShape = new PolygonShape();
		
		
		// draw the shape of the chassis, must be counterclockwise
		chassisShape.set(new float[] { -width / 2, -height / 2, // bottom left
										
										width / 2 , -height / 2, // bottom right
										
										width / 2 + .2f, 0,
										
										width / 2, height / 2,	// top right

										-width / 2 - .2f, height / 2	// top left
										});
		fixtureDef.shape = chassisShape;
		
		chassis = world.createBody(bodyDef);
		chassis.createFixture(fixtureDef);
		
		// create the sled
		PolygonShape sledShape = new PolygonShape();
		sledShape.set(new float[] { 0, -height/2, 	// lower left corner
									width / 1.5f, -height/2 ,		// right edge
									0, height/2 - .2f		// top left
		});
		sledFixtureDef.shape = sledShape;
		sled = world.createBody(bodyDef);
		sled.createFixture(sledFixtureDef);
		
		// create a joint between the chassis and the sled
		RevoluteJointDef sledJointDef = new RevoluteJointDef();
		sledJointDef.bodyA = chassis;
		sledJointDef.bodyB = sled;
		sledJointDef.collideConnected = false;
		sledJointDef.localAnchorA.set(new Vector2(width/2 + .3f, 0));
		sledJointDef.enableLimit = true;
		sledJointDef.maxMotorTorque = 3000.0f;
		sledJointDef.lowerAngle = (float) Math.toRadians(-5);
		sledJointDef.upperAngle = (float) Math.toRadians(45);
		sledJoint = (RevoluteJoint)world.createJoint(sledJointDef);
		
		
		// create the wheels
		CircleShape wheelShape = new CircleShape();
		wheelShape.setRadius(height / 3.5f);
		Sprite wheelSprite = new Sprite(new Texture("data/img/car/wheel.png"));
		wheelSprite.setSize(wheelShape.getRadius() * 2, wheelShape.getRadius() * 2);
		wheelSprite.setOrigin(wheelSprite.getWidth() / 2, wheelSprite.getHeight() / 2);
		wheelFixtureDef.shape = wheelShape;
		
		leftWheel = world.createBody(bodyDef);
		leftWheel.createFixture(wheelFixtureDef);
		leftWheel.setUserData(wheelSprite);
		
		rightWheel = world.createBody(bodyDef);
		rightWheel.createFixture(wheelFixtureDef);
		rightWheel.setUserData(wheelSprite);
		
		
		// create the axis'
		WheelJointDef axisDef = new WheelJointDef();
		axisDef.bodyA = chassis;
		axisDef.localAxisA.set(Vector2.Y);
		axisDef.frequencyHz = fixtureDef.density;
		axisDef.maxMotorTorque = motorTorque;
		
		axisDef.bodyB = leftWheel;
		axisDef.localAnchorA.set(-width / 2 + wheelShape.getRadius(), -height / 2);
		
		leftAxis = (WheelJoint) world.createJoint(axisDef);
		
		axisDef.bodyB = rightWheel;
		axisDef.localAnchorA.x *= -1;
		rightAxis = (WheelJoint) world.createJoint(axisDef);
		
		// radar
		CircleShape radarDishShape = new CircleShape();
		radarDishShape.setRadius(4);
		radarFixtureDef.shape = radarDishShape;
		
		Body tower;
		FixtureDef towerDef =  new FixtureDef();
		FixtureDef dishShapeDef = new FixtureDef();
		towerDef = radarFixtureDef;
		PolygonShape towerShape = new PolygonShape();
		towerShape.setAsBox(.25f, 1f, new Vector2(0,1f),
							0f	// rotation
		);
		
		dishShapeDef = towerDef;
		
		CircleShape dishShape = new CircleShape();
		dishShape.setRadius(.25f);
		
		towerDef.shape = towerShape;
//		tower = world.createBody(bodyDef);
//		tower.createFixture(towerDef);
		
		// create the shape for the sensing area
		float radius = 8;
		float angle;
		Vector2[] vertices =  new Vector2[8];
		vertices[0] = new Vector2(0,0);
		for (int index = 0; index < 7; index++ ) {
			angle = (float) Math.toRadians(index / 6.0 * 90);
			vertices[index+1] = new Vector2((float)(radius * Math.cos(angle)), 
											(float)(radius * Math.sin(angle)));
		}
		PolygonShape radarSenseArea = new PolygonShape();
		radarSenseArea.set(vertices);
		towerDef.shape = radarSenseArea;
//		tower.createFixture(towerDef);
		
		// exhaust
		exhaust = new ParticleEmitter();
		try {
			exhaust.load(Gdx.files.internal("data/exhaust").reader(2024));
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		Sprite particle = new Sprite(new Texture("data/particle.png"));
		exhaust.setSprite(particle);
		exhaust.getScale().setHigh(0.3f);
		exhaust.start();
	}
	public void render(SpriteBatch spriteBatch, Camera camera, float delta) {
		// update speed 
		speed = (1 - speedDecay) * speed + (speedDecay * chassis.getLinearVelocity().x);
		x = chassis.getWorldCenter().x;
		y = chassis.getWorldCenter().y;
		exhaust.setPosition(x - width / 2, y); 
		exhaust.start();
		setExhaustRotation();
		exhaust.draw(spriteBatch, delta);
	}
	
	private void raiseSled() {
		sledJoint.enableMotor(true);
		sledJoint.setMotorSpeed(motorSpeed);
	}
	
	private void lowerSled() {
		sledJoint.enableMotor(false);
	}
	
	
	private void setExhaustRotation() {
		float angle = this.chassis.getAngle();
		exhaust.getAngle().setLow(angle - 180);
		exhaust.getAngle().setHighMin(angle - 120);
		exhaust.getAngle().setHighMax(angle - 240);		
	}
	
	@Override
	public boolean keyDown(int keycode) {
			
		switch(keycode) {
			// jump
			case Keys.W:
				break;
			// move right
			case Keys.A:
				// disable brakes
				rightAxis.enableMotor(false);
				leftAxis.enableMotor(true);
				leftAxis.setMotorSpeed(motorSpeed);
				break;
			// move down
			case Keys.S:
				leftAxis.enableMotor(true);
				leftAxis.setMotorSpeed(0);
				rightAxis.enableMotor(true);
				rightAxis.setMotorSpeed(0);
				break;
			// move right
			case Keys.D:
				// disable brakes
				rightAxis.enableMotor(false);
				leftAxis.enableMotor(true);
				leftAxis.setMotorSpeed(-motorSpeed);
				break;

			case Keys.H:
				holdMotors = !holdMotors;
				break;
			// spin clockwise
			case Keys.Q:
				//boxBody.applyAngularImpulse(1, false);
				break;
			// spin counterclockwise
			case Keys.E:
				//boxBody.applyAngularImpulse(-1, false);
				break;
			case Keys.SPACE:
				
				raiseSled();
				/**
				// create a bullet
				bodyDef.bullet = true;	// allows for more velocity/position calculations per second
				//bodyDef.position.set(boxBody.getWorldCenter().add(new Vector2(0,1)));
				//vector = MyMath.getRectCoords(10, boxBody.getAngle());
				shape =  new CircleShape();
				shape.setRadius(.1f);
				fixtureDef.shape = shape;
				
				// add the bullet to the world
				bullet = world.createBody(bodyDef);
				// add the fixture to the bullet
				bullet.createFixture(fixtureDef);
				// propel the bullet
				bullet.applyLinearImpulse(vector, bullet.getWorldCenter(), false);
				// reset the body def bullet value
				bodyDef.bullet = false;
				/**/
				break;
			case Keys.CONTROL_LEFT:
				lowerSled();
				break;
		}
		return super.keyDown(keycode);
	}
	
	@Override
	public boolean keyUp(int keycode) {
		// TODO Auto-generated method stub
		return super.keyUp(keycode);
	}

	public Body getChassis() {
		return chassis;
	}

	@Override
	public Body getBody() {
		return chassis;
	}


	@Override
	public ArrayList<Body> getBodies() {
		ArrayList<Body> bodies = new ArrayList<Body>();
		bodies.add(chassis);
		bodies.add(leftWheel);
		bodies.add(rightWheel);
		bodies.add(sled);
		return bodies;
	}

	@Override
	public ArrayList<String> getStats() {
		ArrayList<String> stats = new ArrayList<String>();
		stats.add("X:" + StringHelper.getDecimalFormat(chassis.getPosition().x, 0));
		stats.add("Speed: " + StringHelper.getDecimalFormat(speed, 3));
		return stats;
	}
	@Override
	public void sendHome() {
		// TODO Auto-generated method stub
		
	}
	@Override
	public Table getLearningMenu() {
		// TODO Auto-generated method stub
		return null;
	}
	@Override
	public float getUpdateTimer() {
		// TODO Auto-generated method stub
		return 0;
	}
	@Override
	public void setUpdateTimer(float updateTimer) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void setManualControl(boolean isManuControl) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public boolean getManualControl() {
		// TODO Auto-generated method stub
		return false;
	}
	@Override
	public float getMinLearningRate() {
		// TODO Auto-generated method stub
		return 0;
	}
	@Override
	public float getMaxLearningRate() {
		// TODO Auto-generated method stub
		return 0;
	}
	@Override
	public void setMinLearningRate(float minLearningRate) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void setMaxLearningRate(float maxLearningRate) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public float getRandomness() {
		// TODO Auto-generated method stub
		return 0;
	}
	@Override
	public void setRandomness(float randomness) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public float getMinRandomness() {
		// TODO Auto-generated method stub
		return 0;
	}
	@Override
	public float getMaxRandomness() {
		// TODO Auto-generated method stub
		return 0;
	}
	@Override
	public void setMinRandomness(float minRandomness) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void setMaxRandomness(float maxRandomness) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public void setLearningRate(float learningRate) {
		// TODO Auto-generated method stub
		
	}
	@Override
	public float getLearningRate() {
		// TODO Auto-generated method stub
		return 0;
	}


}
