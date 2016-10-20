package com.alec.walker.Models;

import java.util.ArrayList;
import java.util.HashMap;

import com.alec.Assets;
import com.alec.walker.StringHelper;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input.Keys;
import com.badlogic.gdx.InputAdapter;
import com.badlogic.gdx.graphics.Camera;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.ParticleEmitter;
import com.badlogic.gdx.graphics.g2d.Sprite;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.physics.box2d.Body;
import com.badlogic.gdx.physics.box2d.BodyDef;
import com.badlogic.gdx.physics.box2d.BodyDef.BodyType;
import com.badlogic.gdx.physics.box2d.CircleShape;
import com.badlogic.gdx.physics.box2d.FixtureDef;
import com.badlogic.gdx.physics.box2d.PolygonShape;
import com.badlogic.gdx.physics.box2d.World;
import com.badlogic.gdx.physics.box2d.joints.RevoluteJoint;
import com.badlogic.gdx.physics.box2d.joints.RevoluteJointDef;
import com.badlogic.gdx.physics.box2d.joints.WheelJoint;
import com.badlogic.gdx.physics.box2d.joints.WheelJointDef;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.ui.CheckBox;
import com.badlogic.gdx.scenes.scene2d.ui.Label;
import com.badlogic.gdx.scenes.scene2d.ui.Slider;
import com.badlogic.gdx.scenes.scene2d.ui.Table;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;

public class LeggedCrate extends BasicAgent {
	
	private Body						body, rightArm, rightWrist,
										rightFoot, leftArm, leftWrist,
										leftFoot, leftWheel, rightWheel;
	private RevoluteJoint				leftArmJoint, leftWristJoint, rightArmJoint,
										rightWristJoint;

	private WheelJoint					leftAxis, rightAxis;
	private FixtureDef					fixturDef;
	private ParticleEmitter				wristTrail, armTrail, bodyTrail;

	private boolean						isPlayer;
	private float						armTorque;
	private float						wristTorque;
	private int							leftArmLowerLimit;
	private int							rightArmLowerLimit;
	private int							armUpperLimit;
	private int							leftWristLowerLimit;
	private int							leftWristUpperLimit;
	private int							rightWristLowerLimit;
	private int							actionCount;
	private float						armSpeed;
	private float						wristSpeed;
	private int							armRange;
	private int							wristRange;
	private float						armLength;
	private float						wristLength;

	private float						x, y;
	private float						width, height;

	private float						speed;
	private float						speedDecay;

	private boolean						holdMotors		= true;
	private float						previousSpeed;
	private float						acceleration;
	private float						accelerationDecay;
	private float						maxSpeed;

	private float[][][][][]				QValues;

	private HashMap<String, Float>		qFunctionWeights;
	private HashMap<String, Float[]>	QValuesMap;
	private int							previousAction;
	private String						previousState;
	private float						qLearningRate;
	private float						randomness;
	private float						minRandomness;
	private float						maxRandomness;
	private float						minLearningRate;
	private float						maxLearningRate;
	private boolean						isLearning;
	private int							previousRightArmAngle, previousRightWristAngle,
										previousRightAction;
	private int							previousLeftArmAngle, previousLeftWristAngle,
										previousLeftAction;
	private float						previousValue;
	private float						expectedValue;
	private float						bestValue;
	private float						worstValue;
	private float						error;
	private float						maxX;
	private float						maxY;
	private float						previousY;
	private float						previousX;
	private float						angle;
	private float						previousAngle;
	private float						deltaAngle;
	private float						deltaY;
	private float						deltaX;
	private float						initY;
	private float						timeSinceGoodValue;
	private float						accumulator		= 0;
	private float						valueDelta;
	private float						valueVelocity;
	private float						impatience;
	private int							mainGoal;
	private int							goal;
	private int							goals;

	private float						updateTimer		= 0;
	private float						updateTimeout	= 0.0000010f;
	private float						speedWeight, deltaWeight, accelerationWeight, heightWeight;

	private Slider				sldMinRandomness, sldMaxRandomness, sldMinLearningRate,
	sldMaxLearningRate;

	
	public LeggedCrate(World world,
			float x, float y,
			float width, float height) {
		this.x = x;
		this.y = y;
		this.initY = y;
		this.previousY = y;
		this.deltaY = 0;
		this.previousX = x;
		this.deltaX = 0;
		this.maxX = -100;
		this.maxY = -100;
		this.width = width;
		this.height = height;
		this.speed = 0;
		this.speedDecay = 0.1f;
		acceleration = 0;
		accelerationDecay = 0.01f;
		this.isLearning = false;
		error = 0;
		this.isPlayer = false;

		this.actionCount = 12;

		// arm properties
		armLength = 1.55f * height;
		wristLength = armLength * 1.2f;
		armSpeed = .2f;
		wristSpeed = .2f;
		leftArmLowerLimit = 90;
		leftWristLowerLimit = -210;
		rightArmLowerLimit = -30;
		rightWristLowerLimit = 90;
		wristRange = 300;
		armRange = 120;

		// init motor torques
		armTorque = 5000.0f;
		wristTorque = 5000.0f;

		BodyDef bodyDef = new BodyDef();
		bodyDef.type = BodyType.DynamicBody;
		bodyDef.position.set(x, y);
		bodyDef.linearDamping = 0.05f;

		FixtureDef fixtureDef = new FixtureDef();

		fixtureDef.density = 1f;
		// fixtureDef.friction = .001f;
		fixtureDef.friction = .01f;
		fixtureDef.restitution = 0.0f;

		// create the chassis
		PolygonShape chassisShape = new PolygonShape();

		// draw the shape of the chassis, must be counterclockwise
		chassisShape.set(new float[] { -width / 2, -height / 2, // bottom left

				width / 2, -height / 2, // bottom right

				width / 2 + .2f, 0,

				width / 2, height / 2,	// top right

				-width / 2 - .2f, height / 2	// top left
		});
		fixtureDef.shape = chassisShape;

		body = world.createBody(bodyDef);
		body.createFixture(fixtureDef);

		bodyDef.linearDamping = 0.0f;

		PolygonShape armShape = new PolygonShape();
		// armShape.setAsBox(armLength, .1f * height);
		armShape.set(new float[] { armLength, 0, // bottom left

				-armLength, 0, // bottom right

				-armLength, -armLength * .235f,
				-armLength, armLength * .25f
		});
		fixtureDef.shape = armShape;
		fixtureDef.density = 1f;
		rightArm = world.createBody(bodyDef);
		rightArm.createFixture(fixtureDef);
		leftArm = world.createBody(bodyDef);
		leftArm.createFixture(fixtureDef);

		// create left
		RevoluteJointDef leftArmJointDef = new RevoluteJointDef();
		leftArmJointDef.bodyA = body;
		leftArmJointDef.bodyB = leftArm;
		leftArmJointDef.collideConnected = false;
		leftArmJointDef.localAnchorA.set(new Vector2(-(width / 2), (height / 2)));
		leftArmJointDef.localAnchorB.set(new Vector2((-armLength), 0));
		leftArmJointDef.enableLimit = true;
		leftArmJointDef.maxMotorTorque = armTorque;
		leftArmJointDef.lowerAngle = (float) Math.toRadians(leftArmLowerLimit);
		leftArmJointDef.upperAngle = (float) Math.toRadians(leftArmLowerLimit + armRange);
		leftArmJoint = (RevoluteJoint) world.createJoint(leftArmJointDef);

		RevoluteJointDef rightArmJointDef = new RevoluteJointDef();
		rightArmJointDef.bodyA = body;
		rightArmJointDef.bodyB = rightArm;
		rightArmJointDef.collideConnected = false;
		rightArmJointDef.localAnchorA.set(new Vector2((width / 2), (height / 2)));
		rightArmJointDef.localAnchorB.set(new Vector2((-armLength), 0));
		rightArmJointDef.enableLimit = true;
		rightArmJointDef.maxMotorTorque = armTorque;
		rightArmJointDef.lowerAngle = (float) Math.toRadians(rightArmLowerLimit);
		rightArmJointDef.upperAngle = (float) Math.toRadians(rightArmLowerLimit + armRange);
		rightArmJoint = (RevoluteJoint) world.createJoint(rightArmJointDef);

		PolygonShape wristShape = new PolygonShape();

		wristShape.set(new float[] { 0, 0, // bottom left

				-wristLength * .25f, wristLength * 1.25f, // bottom right

				-wristLength * .25f, wristLength * .25f
		});
		fixtureDef.shape = wristShape;
		fixtureDef.density = 1f;
		fixtureDef.restitution = 0.00f;
		// fixtureDef.friction = .85f;
		fixtureDef.friction = .32f;
		rightWrist = world.createBody(bodyDef);
		rightWrist.createFixture(fixtureDef);

		wristShape.set(new float[] { 0, 0, // bottom left

				wristLength * .25f, wristLength * 1.25f, // bottom right

				wristLength * .25f, wristLength * .25f
		});
		fixtureDef.shape = wristShape;
		leftWrist = world.createBody(bodyDef);
		leftWrist.createFixture(fixtureDef);

		// make the wrist joints
		RevoluteJointDef leftWristJointDef = new RevoluteJointDef();
		leftWristJointDef.bodyA = leftArm;
		leftWristJointDef.bodyB = leftWrist;
		leftWristJointDef.collideConnected = false;
		leftWristJointDef.localAnchorA.set(new Vector2(armLength, 0));
		leftWristJointDef.localAnchorB.set(new Vector2(0, 0));
		leftWristJointDef.enableLimit = true;
		leftWristJointDef.maxMotorTorque = wristTorque;
		leftWristJointDef.lowerAngle = (float) Math.toRadians(leftWristLowerLimit);
		leftWristJointDef.upperAngle = (float) Math.toRadians(leftWristLowerLimit + wristRange);
		leftWristJoint = (RevoluteJoint) world.createJoint(leftWristJointDef);

		RevoluteJointDef rightWristJointDef = new RevoluteJointDef();
		rightWristJointDef.bodyA = rightArm;
		rightWristJointDef.bodyB = rightWrist;
		rightWristJointDef.collideConnected = false;
		rightWristJointDef.localAnchorA.set(new Vector2(armLength, 0));
		rightWristJointDef.localAnchorB.set(new Vector2(0, 0));
		rightWristJointDef.enableLimit = true;
		rightWristJointDef.maxMotorTorque = wristTorque;
		rightWristJointDef.lowerAngle = (float) Math.toRadians(rightWristLowerLimit);
		rightWristJointDef.upperAngle = (float) Math.toRadians(rightWristLowerLimit + wristRange);
		rightWristJoint = (RevoluteJoint) world.createJoint(rightWristJointDef);

		FixtureDef wheelFixtureDef = new FixtureDef();
		wheelFixtureDef.density = 1f;
		wheelFixtureDef.friction = 1f;
		wheelFixtureDef.restitution = 0.0000f;

		CircleShape wheelShape = new CircleShape();
		wheelShape.setRadius(height / 5f);
		Sprite wheelSprite = new Sprite(new Texture("data/img/car/wheel.png"));
		wheelSprite.setSize(wheelShape.getRadius() * 2, wheelShape.getRadius() * 2);
		wheelSprite.setOrigin(wheelSprite.getWidth() * .25f, wheelSprite.getHeight() * .5f);
		wheelFixtureDef.shape = wheelShape;

		// leftWheel = world.createBody(bodyDef);
		// wheelFixtureDef.filter.categoryBits = Constants.FILTER_CAR;
		// wheelFixtureDef.filter.maskBits = Constants.FILTER_BOUNDARY;
		// leftWheel.createFixture(wheelFixtureDef);
		// leftWheel.setUserData(wheelSprite);
		// rightWheel = world.createBody(bodyDef);
		// rightWheel.createFixture(wheelFixtureDef);
		// rightWheel.setUserData(wheelSprite);

		// create the axis'
		WheelJointDef axisDef = new WheelJointDef();
		axisDef.bodyA = body;
		axisDef.localAxisA.set(Vector2.Y);
		axisDef.frequencyHz = 10;
		axisDef.maxMotorTorque = 1500.0f;
		axisDef.bodyB = leftWheel;
		axisDef.localAnchorA.set(-(width * 0.5f) + wheelShape.getRadius() * 0.5f,
				-(height * .5f) * 1f);
		// leftAxis = (WheelJoint) world.createJoint(axisDef);
		// leftAxis.enableMotor(false);

		axisDef.bodyB = rightWheel;
		axisDef.localAnchorA.x *= -1;
		// rightAxis = (WheelJoint) world.createJoint(axisDef);
		// rightAxis.enableMotor(false);

		// Particle Emiitters
		Sprite particle = new Sprite(new Texture("data/particle.png"));

		bodyTrail = new ParticleEmitter();
		try {
			bodyTrail.load(Gdx.files.internal("data/trail.fx").reader(2024));
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		bodyTrail.setSprite(particle);
		bodyTrail.getScale().setHigh(.7f);
		bodyTrail.getTint().setActive(true);
		bodyTrail.getGravity().setActive(false);
		bodyTrail.getVelocity().setActive(false);
		float[] colors = new float[] { 0.5f, 0.5f, 0.5f };
		bodyTrail.getTint().setColors(colors);
		bodyTrail.start();

		armTrail = new ParticleEmitter();
		try {
			armTrail.load(Gdx.files.internal("data/trail.fx").reader(2024));
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		armTrail.setSprite(particle);
		armTrail.getScale().setHigh(.2f);
		armTrail.getTint().setActive(true);
		armTrail.getGravity().setActive(false);
		armTrail.getVelocity().setActive(false);
		armTrail.start();

		wristTrail = new ParticleEmitter();
		try {
			wristTrail.load(Gdx.files.internal("data/trail.fx").reader(2024));
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		wristTrail.setSprite(particle);
		wristTrail.getScale().setHigh(.1f);
		wristTrail.getTint().setActive(true);
		wristTrail.getGravity().setActive(false);
		wristTrail.getVelocity().setActive(false);
		wristTrail.start();

		body.setUserData(this);

		isLearning = true;
		initQMap();
	}

	public void initQWeights() {
		qFunctionWeights = new HashMap<String, Float>();

		// left arm
		qFunctionWeights.put("leftArmAngle", 0.0f);
		// left Wrist
		qFunctionWeights.put("leftWristAngle", 0.0f);
		// right arm
		qFunctionWeights.put("rightArmAngle", 0.0f);
		// right Wrist
		qFunctionWeights.put("rightWristAngle", 0.0f);

		// x velocity
		qFunctionWeights.put("xVelocity", 0.0f);
		// x position
		qFunctionWeights.put("xPosition", 0.0f);

		// y velocity
		qFunctionWeights.put("yVelocity", 0.0f);
		// y position
		qFunctionWeights.put("yPosition", 0.0f);

		// angle
		qFunctionWeights.put("bodyAngle", 0.0f);

	}

	public void initQMap() {

		// hold a QValue for
		qLearningRate = 0.001f;
		randomness = 0.02f;
		minRandomness = 0.001f;
		maxRandomness = 0.2f;
		previousAction = 0;
		previousValue = 0;
		expectedValue = 0;
		bestValue = 0;
		worstValue = 0;
		valueDelta = 0;
		valueVelocity = 0;
		timeSinceGoodValue = 0;
		impatience = 0.00001f;
		goal = 0;
		goals = 5;
		mainGoal = 0;
		speedWeight = 1f;
		accelerationWeight = 1f;
		deltaWeight = .1f;
		heightWeight = 1;

		initQWeights();
		QValuesMap = new HashMap<String, Float[]>();
		previousState = "";

		Float[] actionValues = new Float[actionCount];
		for (int actionIndex = 0; actionIndex < actionCount; actionIndex++) {
			actionValues[actionIndex] = 0.0f;
		}

		// init state
		QValuesMap.put(previousState, actionValues);

		this.initY = body.getPosition().y;
		bestValue = initY;
	}

	public void render(SpriteBatch spriteBatch, Camera camera,
			float delta) {

		y = body.getPosition().y;
		deltaY = y - previousY;
		if (y > maxY) {
			maxY = y;
		}

		x = body.getPosition().x;
		deltaX = x - previousX;
		if (x > maxX) {
			maxX = x;
		}

		angle = (int) Math.round(Math.toDegrees(body.getAngle()) * .01f);
		deltaAngle = angle - previousAngle;

		if (isLearning) {
			updateTimer += delta;
			if (updateTimer > updateTimeout) {
				updateTimer = 0.0f;
				QMapUpdate(delta);
			}
		}

		if (body.getPosition().x < -10000 || body.getPosition().x > 10000) {
			body.setTransform(new Vector2(0, 2), 0);
		}

		// Color the trails based on action value
		Color color;
		float value = 0.0f;
		if (previousValue > 0) {
			value = previousValue / bestValue;
			color = new Color(0f, value, 0f, 1);
		} else if (previousValue < 0) {
			value = previousValue / worstValue;
			color = new Color(value, 0f, 0f, 1);
		} else {
			color = Color.WHITE;
		}
		float[] colors = new float[] { color.r, color.g, color.b };

		// armTrail.getTint().setColors(colors);
		// armTrail.getScale().setHigh(value * .1f);
		// armTrail.setPosition(rightArm.getPosition().x, rightArm.getPosition().y);
		// armTrail.draw(spriteBatch, delta);
		//
		// wristTrail.getTint().setColors(colors);
		// wristTrail.getScale().setHigh(value * .1f);
		// wristTrail.setPosition(rightWrist.getWorldCenter().x, rightWrist.getWorldCenter().y);
		// wristTrail.draw(spriteBatch, delta);

		spriteBatch.setProjectionMatrix(camera.combined);
		spriteBatch.begin();
		bodyTrail.getTint().setColors(colors);
		bodyTrail.getScale().setHigh(value * .5f);
		bodyTrail.getScale().setLow(value * .1f);
		bodyTrail.setPosition(body.getWorldCenter().x, body.getWorldCenter().y);
		bodyTrail.draw(spriteBatch, delta);

		 spriteBatch.end();
		previousY = y;
		previousX = x;
		previousAngle = angle;
	}

	public void update(float delta) {

	}

	public Table getMenu() {
		int padding = 10;

		int slideWidth = 1000;

		Table tbl = new Table();
		tbl.row();
		tbl.columnDefaults(0).padRight(padding);
		tbl.columnDefaults(0).padLeft(padding);
		tbl.columnDefaults(0).padTop(padding);
		tbl.columnDefaults(1).padRight(padding);
		tbl.columnDefaults(1).padLeft(padding);
		tbl.columnDefaults(1).padTop(padding + 2);
		tbl.columnDefaults(2).padRight(padding);
		tbl.columnDefaults(2).padLeft(padding);
		tbl.columnDefaults(2).padTop(padding);
		tbl.columnDefaults(3).padRight(padding);
		tbl.columnDefaults(3).padLeft(padding);
		tbl.columnDefaults(3).padTop(padding + 2);
		// tbl.columnDefaults(1).padRight(10);

		// Min Randomness Slide
		tbl.add(new Label("Min Randomness: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		sldMinRandomness = new Slider(0, 1.0f, 0.01f, false, Assets.instance.skin);
		sldMinRandomness.setValue(minRandomness);
		sldMinRandomness.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				minRandomness = sldMinRandomness.getValue();
			}
		});
		tbl.add(sldMinRandomness).width(slideWidth);
		tbl.add(new Label("1", Assets.instance.skin));
		tbl.row();

		// Max Randomness Slide
		tbl.add(new Label("Max Randomness: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		sldMaxRandomness = new Slider(0, 1.0f, 0.01f, false, Assets.instance.skin);
		sldMaxRandomness.setValue(maxRandomness);
		sldMaxRandomness.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				maxRandomness = sldMaxRandomness.getValue();
			}
		});
		tbl.add(sldMaxRandomness).width(slideWidth);
		tbl.add(new Label("1", Assets.instance.skin));
		tbl.row();

		// Arm Speed Slider
		tbl.add(new Label("Arm Speed: ", Assets.instance.skin));
		tbl.add(new Label("0.01", Assets.instance.skin));
		final Slider sldArmSpeed = new Slider(0.01f, 1.0f, 0.01f, false, Assets.instance.skin);
		sldArmSpeed.setValue(armSpeed);
		sldArmSpeed.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				armSpeed = sldArmSpeed.getValue();
			}
		});
		tbl.add(sldArmSpeed).width(slideWidth);
		tbl.add(new Label("1", Assets.instance.skin));
		tbl.row();

		// Wrist Speed Slider
		tbl.add(new Label("Wrist Speed: ", Assets.instance.skin));
		tbl.add(new Label("0.01", Assets.instance.skin));
		final Slider sldWristSpeed = new Slider(0.01f, 1.0f, 0.01f, false, Assets.instance.skin);
		sldWristSpeed.setValue(wristSpeed);
		sldWristSpeed.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				wristSpeed = sldWristSpeed.getValue();
			}
		});
		tbl.add(sldWristSpeed).width(slideWidth);
		tbl.add(new Label("1", Assets.instance.skin));
		tbl.row();

		// isLearning Checkbox
		final CheckBox chbxIsLearning = new CheckBox("isLearning", Assets.instance.skin);
		chbxIsLearning.setChecked(isLearning);
		chbxIsLearning.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				isLearning = chbxIsLearning.isChecked();
			}
		});
		tbl.add(chbxIsLearning);
		tbl.row();

		return tbl;
	}

	
	public float QFunction() {
		float value = 0.0f;
		switch (goal) {

		// Stand Up
			case 0:

				value =
						(body.getLinearVelocity().y * qFunctionWeights.get("yVelocity"))
								+ ((y - maxY) * qFunctionWeights.get("yPosition"))
								+ (angle * qFunctionWeights.get("bodyAngle"));
				break;
		}

		return value;
	}

	public void updateWeights(float currentValue, float previousValue) {

		float delta = currentValue - previousValue;

		if (delta > 0) {

		}

	}

	public void QMapUpdate(float timeDelta) {

		// Get the current state

		float value = 0;

		// Check if should change goal
		if (Math.abs(angle) > 16) {
			goal = 3;
			bestValue = 0;
			worstValue = 0;
		}
		if (goal == 3) {
			if (Math.abs(angle) < 1 || Math.abs(angle) > 35) {
				bestValue = 0;
				worstValue = 0;
				goal = mainGoal;
			}
		}

		switch (goal) {
		// Stand Up
			case 0:
				value = (deltaY * deltaWeight) + (1f * (y - maxY)) + (10f * (y - initY));

				float yVelocity = body.getLinearVelocity().y;
				if (Math.abs(yVelocity) < .001f) {
					yVelocity = 0;
				}

				speed = (1 - speedDecay) * speed + (speedDecay * (yVelocity));
				acceleration = (1 - accelerationDecay) * acceleration
						+ (accelerationDecay * (speed - previousSpeed));

				// check if tipping over
				if (Math.abs(angle) > 30 || Math.abs(angle) > 350) {
					// value -= .5f;
				}

				value = ((y - initY) * heightWeight) + (deltaY * deltaWeight)
						+ ((speed * speedWeight) + (acceleration * accelerationWeight));

				break;
			// Move Forward
			case 1:
				float xVelocity = body.getLinearVelocity().x;
				if (Math.abs(xVelocity) < .01f) {
					xVelocity = 0;
				}

				speed = (1 - speedDecay) * speed + (speedDecay * (xVelocity));
				acceleration = (1 - accelerationDecay) * acceleration
						+ (accelerationDecay * (speed - previousSpeed));

				value = ((speed * speedWeight) + (acceleration * accelerationWeight));

				break;
			// Move Backward
			case 2:
				xVelocity = body.getLinearVelocity().x;
				if (Math.abs(xVelocity) < .01f) {
					xVelocity = 0;
				}

				speed = (1 - speedDecay) * speed + (speedDecay * (xVelocity));
				acceleration = (1 - accelerationDecay) * acceleration
						+ (accelerationDecay * (speed - previousSpeed));
				value = -((speed * 3f) + (acceleration * 0.5f));

				break;

			// Walk Forward
			case 3:
				value = (deltaY * 10) + (.1f * (y - initY)) + (deltaX);
				break;

			// Roll over
			case 4:
				value = (Math.abs(angle) * .01f) + (-deltaAngle * 1000);
				break;
		}

		// value += speed * .01f;

		// if been a while since a good move was taken
		if (timeSinceGoodValue > 5f) {
			// make more random moves
			randomness = Math.min(maxRandomness, randomness *= 1.001f);
			if (timeSinceGoodValue < 100f) {
				// punish
				value -= timeSinceGoodValue * timeSinceGoodValue * impatience;
			}
		} else {
			// make less random moves
			randomness = Math.max(minRandomness, randomness *= .999f);
		}

		// get join angles
		float angleIncrement = 0.1f;
		int leftArmAngle = (int) Math.round(Math.toDegrees(leftArmJoint.getJointAngle())
				* angleIncrement);
		int leftWristAngle = (int) Math
				.round(Math.toDegrees(leftWristJoint.getJointAngle()) * angleIncrement);

		int rightArmAngle = (int) Math.round(Math.toDegrees(rightArmJoint.getJointAngle())
				* angleIncrement);
		int rightWristAngle = (int) Math
				.round(Math.toDegrees(rightWristJoint.getJointAngle()) * angleIncrement);

		int bodyAngle = (int) Math.round(-Math.toDegrees(body.getAngle()) * angleIncrement);

		String state = "" + leftArmAngle + "," + leftWristAngle + "," + rightArmAngle + "," +
				rightWristAngle + "," + goal;

		// find an action to take
		int action = 0;

		// if never been in state
		if (!QValuesMap.containsKey(state)) {
			Float[] actionValues = new Float[actionCount];
			for (int actionIndex = 0; actionIndex < actionCount; actionIndex++) {
				actionValues[actionIndex] = 0.000f;
			}
			// init state
			QValuesMap.put(state, actionValues);
		}

		// search for max qvalue action
		float maxActionQValue = QValuesMap.get(state)[0];
		int maxAction = 0;
		// for each action
		for (int actionIndex = 0; actionIndex < actionCount; actionIndex++) {
			// if action taken
			if (QValuesMap.get(state)[actionIndex] > maxActionQValue) {
				maxActionQValue = QValuesMap.get(state)[actionIndex];
				maxAction = actionIndex;
			}
		}

		// maybe make random move
		if (randomness > Math.random() || maxActionQValue == 0) {
			action = (int) Math.round(Math.random() * (actionCount - 1));
		} else {
			action = maxAction;
		}

		takeAction(action);

		// Evaluate previous action

		// if (!QValuesMap.containsKey(previousState)) {
		// Float[] actionValues = new Float[actionCount];
		// for (int actionIndex = 0; actionIndex < actionCount; actionIndex++) {
		// actionValues[actionIndex] = 0.0f;
		// }
		// for (int actionIndex = 0; actionIndex < actionCount; actionIndex++) {
		// System.out.println("actionValues = " + actionValues[actionIndex]);
		// }
		// // init previousState
		// QValuesMap.put(previousState, actionValues);
		// }

		// if (value > 0) {
		// timeSinceGoodValue = 0.0f;
		// }

		if (value > bestValue) {
			bestValue = value;
		}
		try {
			// Reward
			if (value > (bestValue * .25f)) {
				timeSinceGoodValue = 0.0f;
				QValuesMap.get(previousState)[previousAction] = (1 - qLearningRate)
						* QValuesMap.get(previousState)[previousAction]
						+
						qLearningRate * (100);

				// Punish
			} else if (value < worstValue) {
				timeSinceGoodValue += timeDelta;
				worstValue = value;
				QValuesMap.get(previousState)[previousAction] = (1 - qLearningRate)
						* QValuesMap.get(previousState)[previousAction]
						+
						qLearningRate * (-1 + maxActionQValue);

				// Propagate
			} else {
				timeSinceGoodValue += timeDelta;
				QValuesMap.get(previousState)[previousAction] = (1 - qLearningRate)
						* QValuesMap.get(previousState)[previousAction]
						+
						qLearningRate * (value + maxActionQValue);
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}

		// Remember the reward from the previous action
		expectedValue = maxActionQValue;
		previousValue = value;
		previousAction = action;
		previousState = state;
		error = value - expectedValue;

	}

	public void takeAction(int actionIndex) {
		// take action
		switch (actionIndex) {
			case 0:
				rightArmJoint.enableMotor(true);
				rightArmJoint.setMotorSpeed(0);
				break;
			case 1:
				// move arm down
				rightWristJoint.enableMotor(true);
				rightWristJoint.setMotorSpeed(0);
				break;
			case 2:
				// move wrist left
				rightWristJoint.enableMotor(true);
				rightWristJoint.setMotorSpeed(wristSpeed);
				break;
			case 3:
				// move wrist right
				rightWristJoint.enableMotor(true);
				rightWristJoint.setMotorSpeed(-wristSpeed);
				break;
			case 4:
				// hold wrist right
				rightArmJoint.enableMotor(true);
				rightArmJoint.setMotorSpeed(-armSpeed);
				break;
			case 5:
				// move arm right
				// move arm up
				rightArmJoint.enableMotor(true);
				rightArmJoint.setMotorSpeed(armSpeed);
				break;

			case 6:
				leftArmJoint.enableMotor(true);
				leftArmJoint.setMotorSpeed(0);
				break;
			case 7:
				// move arm down
				leftWristJoint.enableMotor(true);
				leftWristJoint.setMotorSpeed(0);
				break;
			case 8:
				// move wrist left
				leftWristJoint.enableMotor(true);
				leftWristJoint.setMotorSpeed(wristSpeed);
				break;
			case 9:
				// move wrist left
				leftWristJoint.enableMotor(true);
				leftWristJoint.setMotorSpeed(-wristSpeed);
				break;
			case 10:
				// hold wrist left
				leftArmJoint.enableMotor(true);
				leftArmJoint.setMotorSpeed(-armSpeed);
				break;
			case 11:
				// move arm left
				// move arm up
				leftArmJoint.enableMotor(true);
				leftArmJoint.setMotorSpeed(armSpeed);
				break;
		}
	}

	@Override
	public boolean keyDown(int keycode) {

		switch (keycode) {
			case Keys.W:
				leftArmJoint.enableMotor(true);
				leftArmJoint.setMotorSpeed(armSpeed);
				break;
			case Keys.A:
				leftWristJoint.enableMotor(true);
				leftWristJoint.setMotorSpeed(-wristSpeed);
				break;
			case Keys.S:
				leftArmJoint.enableMotor(true);
				leftArmJoint.setMotorSpeed(-armSpeed);
				break;
			case Keys.D:
				leftWristJoint.enableMotor(true);
				leftWristJoint.setMotorSpeed(wristSpeed);
				break;
			case Keys.UP:
				rightArmJoint.enableMotor(true);
				rightArmJoint.setMotorSpeed(-armSpeed);
				break;
			case Keys.LEFT:
				rightWristJoint.enableMotor(true);
				rightWristJoint.setMotorSpeed(-wristSpeed);
				break;
			case Keys.DOWN:
				rightArmJoint.enableMotor(true);
				rightArmJoint.setMotorSpeed(armSpeed);
				break;
			case Keys.RIGHT:
				rightWristJoint.enableMotor(true);
				rightWristJoint.setMotorSpeed(wristSpeed);
				break;
			case Keys.L:
				initQMap();
				isLearning = !isLearning;
				break;
			case Keys.H:
				holdMotors = !holdMotors;
				break;
			case Keys.M:
				bestValue = 0;
				worstValue = 0;
				if (mainGoal + 1 > goals) {
					mainGoal = 0;
				} else {
					mainGoal = mainGoal + 1;
				}

				goal = mainGoal;
				break;

			case Keys.P:
				qLearningRate = Math.min(.99f, qLearningRate += .05f);
				break;
			case Keys.O:
				qLearningRate = Math.max(.001f, qLearningRate -= .05f);
				break;
			case Keys.Q:
				break;
			case Keys.G:
				minRandomness = 0.0f;
				break;
			case Keys.E:
				// minRandomness += 0.01f;
				maxRandomness += 0.01f;
				randomness += 0.1f;
				break;
			case Keys.R:
				// minRandomness -= 0.01f;
				maxRandomness -= 0.01f;
				randomness -= 0.1f;
				break;
			case Keys.SPACE:
				break;
			case Keys.X:
				break;
		}
		return super.keyDown(keycode);
	}

	@Override
	public boolean keyUp(int keycode) {
		switch (keycode) {
			case Keys.W:
				leftArmJoint.setMotorSpeed(0);
				leftArmJoint.enableMotor(holdMotors);
				break;
			case Keys.A:
				leftWristJoint.setMotorSpeed(0);
				leftWristJoint.enableMotor(holdMotors);
				break;
			case Keys.S:
				leftArmJoint.setMotorSpeed(0);
				leftArmJoint.enableMotor(holdMotors);
				break;
			case Keys.D:
				leftWristJoint.setMotorSpeed(0);
				leftWristJoint.enableMotor(holdMotors);
				break;
			case Keys.UP:
				rightArmJoint.setMotorSpeed(0);
				rightArmJoint.enableMotor(holdMotors);
				break;
			case Keys.LEFT:
				rightWristJoint.setMotorSpeed(0);
				rightWristJoint.enableMotor(holdMotors);
				break;
			case Keys.DOWN:
				rightArmJoint.setMotorSpeed(0);
				rightArmJoint.enableMotor(holdMotors);
				break;
			case Keys.RIGHT:
				rightWristJoint.setMotorSpeed(0);
				rightWristJoint.enableMotor(holdMotors);
				break;
		}
		return super.keyUp(keycode);
	}

	public Body getBody() {
		return body;
	}

	@Override
	public ArrayList<Body> getBodies() {
		ArrayList<Body> bodies = new ArrayList<Body>();
		bodies.add(body);
		bodies.add(leftArm);
		// bodies.add(leftFoot);
		bodies.add(leftWrist);
		bodies.add(rightArm);
		// bodies.add(rightFoot);
		bodies.add(rightWrist);
		bodies.add(leftWheel);
		bodies.add(rightWheel);
		return bodies;
	}

	public float getSpeed() {
		return speed;
	}

	public ArrayList<String> getStats() {
		ArrayList<String> stats = new ArrayList<String>();
		stats.add("Learning:" + isLearning);
		stats.add("X:" + StringHelper.getDecimalFormat(body.getPosition().x, 0));
		stats.add("deltaX:" + StringHelper.getDecimalFormat(deltaX, 1));
		stats.add("maxX:" + StringHelper.getDecimalFormat(maxX, 1));
		stats.add("Y:" + StringHelper.getDecimalFormat(y - initY, 0));
		stats.add("deltaY:" + StringHelper.getDecimalFormat(deltaY, 1));
		stats.add("maxY:" + StringHelper.getDecimalFormat(maxY, 1));
		stats.add("yVelocity:" + StringHelper.getDecimalFormat(body.getLinearVelocity().y, 1));
		stats.add("Speed: " + StringHelper.getDecimalFormat(speed, 3));
		stats.add("Acceleration: " + StringHelper.getDecimalFormat(acceleration, 3));
		stats.add("Angle: "
				+ StringHelper.getDecimalFormat(
						previousAngle, 3));
		// stats.add("absAngle: "
		// + StringHelper.getDecimalFormat(
		// Math.abs(previousAngle), 3));
		stats.add("deltaAngle: "
				+ StringHelper.getDecimalFormat(
						deltaAngle, 3));
		stats.add("previousValue:" + StringHelper.getDecimalFormat(previousValue, 0));
		stats.add("previousAction:" + previousAction);
		if (previousState != null) {
			stats.add("previousState: " + previousState);
		}
		// stats.add("valueDelta:" + StringHelper.getDecimalFormat(valueDelta, 1));
		// stats.add("valueVelocity:" + StringHelper.getDecimalFormat(valueVelocity, 1));
		stats.add("timeSinceGoodValue:" + StringHelper.getDecimalFormat(timeSinceGoodValue, 1));
		stats.add("bestValue:" + StringHelper.getDecimalFormat(bestValue, 2));
		stats.add("worstValue:" + StringHelper.getDecimalFormat(worstValue, 2));
		stats.add("E:" + StringHelper.getDecimalFormat(randomness, 3));
		stats.add("Alpha:" + StringHelper.getDecimalFormat(qLearningRate, 4));
		stats.add("mainGoal:" + mainGoal);
		stats.add("goal:" + goal);
		stats.add("error:" + error);
		stats.add("expectedValue:" + expectedValue);

		return stats;

	}

	public boolean isPlayer() {
		return isPlayer;
	}





}
