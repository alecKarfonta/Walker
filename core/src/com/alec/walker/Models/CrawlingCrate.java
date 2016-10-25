package com.alec.walker.Models;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import org.joda.time.Duration;

import com.alec.Assets;
import com.alec.walker.Constants;
import com.alec.walker.GamePreferences;
import com.alec.walker.MyMath;
import com.alec.walker.StringHelper;
import com.alec.walker.Views.Play;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input.Keys;
import com.badlogic.gdx.graphics.Camera;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.Texture.TextureFilter;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.ParticleEmitter;
import com.badlogic.gdx.graphics.g2d.Sprite;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.math.MathUtils;
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
import com.badlogic.gdx.scenes.scene2d.ui.Label;
import com.badlogic.gdx.scenes.scene2d.ui.Slider;
import com.badlogic.gdx.scenes.scene2d.ui.Table;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;

public class CrawlingCrate extends BasicAgent {
	private static final String	TAG			= CrawlingCrate.class.getName();

	// Body parts
	private Play				play;
	private Body				body, arm, wrist,
								leftWheel, rightWheel;
	private PolygonShape		chassisShape;
	// Joints: arm and wheels
	private RevoluteJoint		armJoint, wristJoint;
	private WheelJoint			leftAxis, rightAxis;

	// Particle emitter for visualizing reward signal
	private ParticleEmitter		bodyTrail;
	private ParticleEmitter		finishParticleEmitter;
	private BitmapFont			font;
	// Size
	private float				width, height;
	private float[]				bodyShape;
	private float				density;

	// Position
	// Arm parameters
	private float				armWidth;
	private float				wristWidth;
	private float				armLength;
	private float				wristLength;
	private float				armSpeed;
	private float				wristSpeed;
	private int					armRange;
	private int					wristRange;
	private float				armTorque;
	private float				wristTorque;

	// Body Properties
	private float				legSpread;
	private float				wheelRadius;
	private float				suspension;
	private float				rideHeight;

	// Performance stats
	private float				speed;
	private float				maxSpeed;
	private float				speedDecay;
	private float				previousSpeed;
	private float				acceleration;

	public float[][][]			QValues;
	public int[][][]			SACounts;
	private int					previousArmAngle, previousWristAngle;
	private float				previousReward, previousExplorationValue;
	private float				oldValue, newValue;
	private float				timeSinceGoodValue;

	public int					rank;
	public int					finishLine;

	// State is each state property
	private float[]				state;
	// Save a weight for each feature
	private float[]				weights;
	private float[]				previousState;

	// QFunction Weights
	private float				speedValueWeight, averageSpeedValueWeight;

	// Control the reduction in state space
	private float				precision;

	private boolean				holdMotors	= true;
	public boolean				showName	= false;

	public CrawlingCrate(Play play) {
		this.play = play;

		rank = 0;
		finishLine = 1500;
	}

	public void init(World world, float x, float y,
			float width, float height) {
		speedValueWeight = GamePreferences.instance.speedValueWeight;
		averageSpeedValueWeight = (float) Math.random() * 10;

		learningRate = GamePreferences.instance.learningRate;

		armSpeed = GamePreferences.instance.armSpeed;
		wristSpeed = GamePreferences.instance.wristSpeed;

		armRange = GamePreferences.instance.armRange;
		wristRange = GamePreferences.instance.wristRange;

		// init motor torques
		armTorque = GamePreferences.instance.armTorque;
		wristTorque = GamePreferences.instance.wristTorque;
		futureDiscount = GamePreferences.instance.futureDiscount;

		float density = GamePreferences.instance.density;
		float updateTimer = GamePreferences.instance.updateTimer;

		init(world, x, y, width, height, randomness, learningRate, minRandomness, maxRandomness,
				minLearningRate, maxLearningRate, learningRateDecay, updateTimer, futureDiscount,
				speedValueWeight,
				averageSpeedValueWeight,
				armSpeed, wristSpeed, armRange, wristRange, armTorque, wristTorque, bodyShape,
				density);
	}

	public void init(
			World world,
			float x, float y,
			float width, float height,
			float randomness, float qlearningRate,
			float minRandomness, float maxRandomness,
			float minlearningRate, float maxLearningRate,
			float learningRateDecay,
			float updateTimer,
			float futureDiscount,
			float speedValueWeight, float averageSpeedValueWeight,
			float armSpeed, float wristSpeed,
			int armRange, int wristRange,
			float armTorque, float wristTorque,
			float[] bodyShape, float density) {
		float defaultLegSpread = 0.5f;
		float defaultWheelRadius = height / 3.5f;
		float defaultSuspension = GamePreferences.instance.suspension;
		float rideHeight = 1;
		float armWidth = .2f * height;
		float armLength = 1.5f * height;
		float wristLength = armLength * 1.5f;
		float wristWidth = -wristLength * .1f;
		// Calculate arm lengths based on body size

		init(world, x, y, width, height, randomness, qlearningRate, minRandomness, maxRandomness,
				minlearningRate, maxLearningRate, learningRateDecay, updateTimer, futureDiscount,
				speedValueWeight, averageSpeedValueWeight, armSpeed, wristSpeed, armRange,
				wristRange, armTorque, wristTorque, bodyShape, density, defaultLegSpread,
				defaultWheelRadius, defaultSuspension, rideHeight, armLength, armWidth, wristLength, wristWidth);
	}

	public void init(
			World world,
			float x, float y,
			float width, float height,
			float randomness, float qlearningRate,
			float minRandomness, float maxRandomness,
			float minlearningRate, float maxLearningRate,
			float learningRateDecay,
			float updateTimer,
			float futureDiscount,
			float speedValueWeight, float averageSpeedValueWeight,
			float armSpeed, float wristSpeed,
			int armRange, int wristRange,
			float armTorque, float wristTorque,
			float[] bodyShape, float density,
			float legSpread, float wheelRadius,
			float suspension, float rideHeight,
			float armLength, float armWidth,
			float wristLength, float wristWidth

			) {
		if (isDebug) {
			System.out.println("CrawlingCrate.init(world," + x + ", " + y + ", " + width + ", "
					+ height + ", " + randomness + ", " +
					qlearningRate + ", " + minlearningRate + ", " + maxLearningRate + ", "
					+ minlearningRate + ", " + maxLearningRate + ", "
					+ learningRateDecay + ", " + speedValueWeight + ", " + averageSpeedValueWeight
					+ ", " + armSpeed + ", "
					+ wristSpeed + ", " + armRange + ", " + wristRange + ", " + armTorque + ", "
					+ wristTorque + ", bodyShape, " + density);
		}
		this.width = width;
		this.height = height;
		this.randomness = randomness;
		this.learningRate = qlearningRate;
		this.minLearningRate = minlearningRate;
		this.maxLearningRate = maxLearningRate;
		this.learningRateDecay = learningRateDecay;
		this.futureDiscount = futureDiscount;
		this.speedValueWeight = speedValueWeight;
		this.averageSpeedValueWeight = averageSpeedValueWeight;
		this.wristLength = wristLength;
		this.wristWidth = wristWidth;
		this.armLength = armLength;
		this.armWidth = armWidth;
		this.armRange = armRange;
		this.armSpeed = armSpeed;
		this.wristSpeed = wristSpeed;
		this.armTorque = armTorque;
		this.wristRange = wristRange;
		this.wristTorque = wristTorque;
		this.density = density;
		this.updateTimer = updateTimer;
		this.suspension = suspension;
		this.legSpread = legSpread;
		this.wheelRadius = wheelRadius;
		this.impatience = GamePreferences.instance.impatience;
		this.precision = 0.1f;

		// Calculate body shape
		bodyShape = new float[] { -((width / 2) * .95f), -height / 2, // bottom left

				width / 2, -height / 2, // bottom right

				width / 2 + .2f, 0,

				width / 2 * 1.25f, (height / 2) * 1.1f,	// top right

				-width / 2 - .2f, height / 2,	// top left

				-((width / 2) * .95f), -height / 2 // bottom left
		};

		this.bodyShape = bodyShape;

		// Init from super without randomness
		super.init(false);

		// Init some local variables
		isManualControl = false;
		this.goal = 0;
		this.goals = new String[] { "Move Right", "Move Left", "Go Home" };
		this.speed = 0;
		this.speedDecay = 0.8f;
		acceleration = 0;
		state = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, };
		previousState = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, };


		initLearning();

		// Define body properties
		BodyDef bodyDef = new BodyDef();
		bodyDef.type = BodyType.DynamicBody;
		bodyDef.position.set(x, y);
		bodyDef.linearDamping = GamePreferences.instance.linearDampening;
		bodyDef.allowSleep = false;
		bodyDef.bullet = true;

		FixtureDef fixtureDef = new FixtureDef();

		fixtureDef.density = density;
		fixtureDef.friction = .01f;
		fixtureDef.restitution = .01f;
		fixtureDef.filter.categoryBits = Constants.FILTER_CAR;
		fixtureDef.filter.maskBits = Constants.FILTER_BOUNDARY;

		// create the chassis
		chassisShape = new PolygonShape();

		// Set the chassis shape
		chassisShape.set(bodyShape);
		fixtureDef.shape = chassisShape;

		// Create the chassis body
		body = world.createBody(bodyDef);
		body.createFixture(fixtureDef);

		// Turn off linear dampening for rest of body
		bodyDef.linearDamping = 0.0f;

		PolygonShape armShape = new PolygonShape();
		armShape.setAsBox(armLength, armWidth);
		fixtureDef.shape = armShape;
		fixtureDef.density = density * .15f;
		fixtureDef.filter.categoryBits = Constants.FILTER_CAR;
		fixtureDef.filter.maskBits = Constants.FILTER_BOUNDARY;

		arm = world.createBody(bodyDef);
		arm.createFixture(fixtureDef);

		RevoluteJointDef armJointDef = new RevoluteJointDef();
		armJointDef.bodyA = body;
		armJointDef.bodyB = arm;
		armJointDef.collideConnected = false;
		armJointDef.localAnchorA.set(new Vector2(0, (.1f * height) * .5f - (height * 0)));
		armJointDef.localAnchorB.set(new Vector2((-armLength), 0));
		armJointDef.enableLimit = true;
		armJointDef.maxMotorTorque = armTorque;
		armJointDef.lowerAngle = 0;
		armJointDef.upperAngle = (float) Math.toRadians(armRange);
		armJoint = (RevoluteJoint) world.createJoint(armJointDef);
		armJoint.enableMotor(true);

		PolygonShape wristShape = new PolygonShape();

		wristShape.set(new float[] { -wristWidth, 0, // bottom left

				wristWidth, 0, // bottom right

				wristWidth, wristLength
		});
		// wristShape.setAsBox(wristLength, .05f * height);
		fixtureDef.shape = wristShape;
		fixtureDef.friction = GamePreferences.instance.friction;
		fixtureDef.restitution = 0f;

		wrist = world.createBody(bodyDef);
		wrist.createFixture(fixtureDef);
		// wrist.setBullet(true);

		RevoluteJointDef wristJointDef = new RevoluteJointDef();
		wristJointDef.bodyA = arm;
		wristJointDef.bodyB = wrist;
		wristJointDef.collideConnected = false;
		wristJointDef.localAnchorA.set(new Vector2(armLength, 0));
		// rightWristJointDef.localAnchorB.set(new Vector2(-wristLength * 1.1f, 0));
		wristJointDef.enableLimit = true;
		wristJointDef.maxMotorTorque = wristTorque;
		wristJointDef.referenceAngle = (float) Math.toRadians(110);
		wristJointDef.lowerAngle = 0;
		wristJointDef.upperAngle = (float) Math.toRadians(wristRange);
		wristJoint = (RevoluteJoint) world.createJoint(wristJointDef);
		wristJoint.enableMotor(true);

		// Wheel fixture def
		FixtureDef wheelFixtureDef = new FixtureDef();
		wheelFixtureDef.density = density;
		wheelFixtureDef.friction = .15f;
		wheelFixtureDef.restitution = 0;
		wheelFixtureDef.filter.categoryBits = Constants.FILTER_CAR;
		wheelFixtureDef.filter.maskBits = Constants.FILTER_BOUNDARY;

		// create the wheels
		CircleShape wheelShape = new CircleShape();
		wheelShape.setRadius(wheelRadius);
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
		axisDef.bodyA = body;
		axisDef.localAxisA.set(Vector2.Y);
		axisDef.frequencyHz = suspension;
		axisDef.maxMotorTorque = 15.0f;

		axisDef.bodyB = leftWheel;
		axisDef.localAnchorA.set(-width * legSpread + wheelShape.getRadius() * 0.5f,
				-(height * legSpread) - wheelShape.getRadius() * 2);
//		axisDef.localAnchorB.set(0,-rideHeight);

		leftAxis = (WheelJoint) world.createJoint(axisDef);
		leftAxis.enableMotor(false);
		leftAxis.setSpringFrequencyHz(suspension);

		axisDef.bodyB = rightWheel;
		axisDef.localAnchorA.x *= -1;
		rightAxis = (WheelJoint) world.createJoint(axisDef);
		rightAxis.enableMotor(false);
		rightAxis.setSpringFrequencyHz(suspension);

		// Particle Emiitters
		Sprite particle = new Sprite(new Texture("data/particle.png"));

		bodyTrail = new ParticleEmitter();
		try {
			bodyTrail.load(Gdx.files.internal("data/trail.fx").reader(2024));
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		bodyTrail.setSprite(particle);
		bodyTrail.getScale().setHigh(.3f);
		bodyTrail.getTint().setActive(true);
		bodyTrail.getGravity().setActive(false);
		bodyTrail.getVelocity().setActive(false);
		float[] colors = new float[] { 0.5f, 0.5f, 0.5f };
		bodyTrail.getTint().setColors(colors);
		bodyTrail.start();

		finishParticleEmitter = new ParticleEmitter();
		try {
			finishParticleEmitter.load(Gdx.files.internal("data/wall.fx").reader(2024));
		} catch (Exception ex) {
			ex.printStackTrace();
		}
		finishParticleEmitter.setSprite(particle);
//		finishParticleEmitter.getScale().setHigh(.5f);
		
		finishParticleEmitter.getTint().setActive(true);
		finishParticleEmitter.getGravity().setActive(false);
		finishParticleEmitter.getVelocity().setActive(false);
//		finishParticleEmitter.getTint().setColors(new float[] { 0.0f, 1f, 0f });
		finishParticleEmitter.start();
		
		// Set body's user data to this so it can be referenced from game
		body.setUserData(this);

	}

	public void initLearning() {
		// System.out.println("initLearning()");
		age = new Duration(0);
		actionCount = 6;

		previousAction = 0;
		previousValue = 0;
		previousVelocity = 0;
		bestValue = 0;
		worstValue = 0;
		valueDelta = 0;
		valueVelocity = 0;
		timeSinceGoodValue = 0;

		// System.out.println("initLearning() : armRange = " + armRange);
		// System.out.println("initLearning(): wristRange = " + wristRange);

		int QArmRange = (int) ((armRange + 1) * precision) + 1;
		int QWristRange = (int) ((wristRange + 1) * precision) + 1;

		// System.out.println("initLearning() : QArmRange = " + QArmRange);
		// System.out.println("initLearning(): QWristRange = " + QWristRange);

		// Init Q Table to all 0's
		QValues = new float[QArmRange][QWristRange][actionCount];

		for (int armIndex = 0; armIndex < QArmRange; armIndex++) {
			for (int wristIndex = 0; wristIndex < QWristRange; wristIndex++) {
				for (int actionIndex = 0; actionIndex < actionCount; actionIndex++) {
					QValues[armIndex][wristIndex][actionIndex] = 0.0f;
				}
			}
		}

		// Init visit counts to all 0's
		SACounts = new int[QArmRange][QWristRange][actionCount];
		for (int x = 0; x < QValues.length; x++) {
			for (int y = 0; y < QValues[x].length; y++) {
				for (int z = 0; z < QValues[x][y].length; z++) {
					SACounts[x][y][z] = 0;
				}
			}
		}

		// Init Q-Function weights
		// Calculate number of weights by adding number of state features and actions
		int weightCount = state.length + 1 + actionCount;
		weights = new float[weightCount];

		// For each weight
		for (int x = 0; x < x; x++) {
			// Init to a small random value
			weights[x] = (float) (0.01f * Math.random());
		}

	}

	public void act(int actionIndex) {
		if (isDebug)
		{
			// System.out.println("act(" + actionIndex + ")");
		}
		// Switch from action index to motor action
		switch (actionIndex) {
			case 0:
				wristJoint.enableMotor(true);
				wristJoint.setMotorSpeed(wristSpeed);
				break;
			case 1:
				wristJoint.enableMotor(true);
				wristJoint.setMotorSpeed(-wristSpeed);
				break;
			case 2:
				armJoint.enableMotor(true);
				armJoint.setMotorSpeed(armSpeed);
				break;
			case 3:
				armJoint.enableMotor(true);
				armJoint.setMotorSpeed(-armSpeed);
				break;
			case 4:
				armJoint.enableMotor(true);
				armJoint.setMotorSpeed(0);
				break;
			case 5:
				// move arm down
				wristJoint.enableMotor(true);
				wristJoint.setMotorSpeed(0);
				break;

		}

		// Save as previous action for next timestep
		previousAction = actionIndex;
		previousActions.add(previousAction);

	}

	// Get the Q Value for the state-action pair
	public float QValue(float[] stateAction) {
		float sum = 0;

		// For each state feature and action
		for (int x = 0; x < stateAction.length; x++) {
			// Add the feature value times the weight to the sum
			sum += stateAction[x] * weights[x];
		}

		return sum;
	}

	public void QFunctionUpdate(float delta) {
		// Get the immediate reward
		float reward = getReward();

		// Save previous state
		previousState = state.clone();

		// Get the current state [armAngle, wristAngle]
		int armAngle = (int) Math.round(
				Math.toDegrees(
						armJoint.getJointAngle()
						)
				);
		armAngle = MathUtils.clamp(armAngle, 0, armRange);

		int wristAngle = (int) Math.round(
				Math.toDegrees(
						wristJoint.getJointAngle())
				);
		wristAngle = MathUtils.clamp(wristAngle, 0, wristRange);

		// Reduce precision of angles
		armAngle = (int) (armAngle * precision);
		wristAngle = (int) (wristAngle * precision);

		// Store state vector, where the action is represented by a one-hot
		state = new float[] { armAngle, wristAngle, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
		// Set previous action index
		state[previousAction + 2] = 1;

		// Get the old value
		oldValue = QValues[previousArmAngle][previousWristAngle][previousAction];

		// Save current value as previous for next step
		previousReward = reward;
		previousArmAngle = armAngle;
		previousWristAngle = wristAngle;
	}

	public float getReward() {
		// Get x velocity of body
		float xVelocity = body.getLinearVelocity().x;

		if (xVelocity > maxSpeed) {
			maxSpeed = xVelocity;
		}

		// If velocity is below threshold
		if (Math.abs(xVelocity) < 0.25f) {
			// Set to zero
			xVelocity = 0f;
			// xVelocity = -(explorationBonus / SACounts[previousArmAngle][previousWristAngle][previousAction]) * .5f;
		} else {
			// boolean isNegative = (xVelocity < 0);
			// Square the velocity
			// xVelocity = xVelocity * xVelocity;
			// if (isNegative) {
			// xVelocity = -1 * xVelocity;
			// }
		}

		// Calculate the speed as a moving average of the velocity
		speed = (1 - speedDecay) * speed + (speedDecay * xVelocity);
		acceleration = speed - previousSpeed;

		// Get the value of the current state
		float reward = (speedValueWeight * xVelocity) + (acceleration * averageSpeedValueWeight);
		// Flip reward
		// reward = -reward;
		// float reward = xVelocity;
		return reward;
	}

	public void getQValue() {

	}

	public void QUpdate(float delta) {

		// Get the immediate reward
		float reward = getReward();

		// Increment the visit count for the previous S, A
		SACounts[previousArmAngle][previousWristAngle][previousAction] += 1;

		// Get the current state [armAngle, wristAngle]
		int armAngle = (int) Math.round(
				Math.toDegrees(
						armJoint.getJointAngle()
						)
				);
		armAngle = MathUtils.clamp(armAngle, 0, armRange);

		int wristAngle = (int) Math.round(
				Math.toDegrees(
						wristJoint.getJointAngle())
				);
		wristAngle = MathUtils.clamp(wristAngle, 0, wristRange);

		// Reduce precision of angles
		armAngle = (int) (armAngle * precision);
		wristAngle = (int) (wristAngle * precision);

		// Find the max value action
		// Init to a random action
		int action = (int) Math.floor(Math.random() * actionCount);
		float actionValue = QValues[armAngle][wristAngle][action];

		try {
			// Start with a random action
			int maxAction = action;
			float maxActionValue = actionValue;
			float explorationBonusValue = 0;
			float maxExplorationBonusValue = 0;

			try {
				// for each action
				for (int actionIndex = 0; actionIndex < actionCount; actionIndex++) {
					// Calculate exploration bonus
					// explorationBonusValue = explorationBonus
					// / (SACounts[armAngle][wristAngle][actionIndex]);
					//
					// if (Float.isInfinite(explorationBonusValue)) {
					// explorationBonusValue = bestValue;
					// }
					//
					// if (explorationBonusValue < 0.01f) {
					// explorationBonusValue = 0;
					// }

					// If value of action is greater than best so far
					// if (((QValues[armAngle][wristAngle][actionIndex]) + explorationBonusValue)
					// > (maxActionValue + maxExplorationBonusValue)) {
					if ((QValues[armAngle][wristAngle][actionIndex]) > maxActionValue) {
						// Save max action value
						maxActionValue = QValues[armAngle][wristAngle][actionIndex];
						// maxActionValue = QValues[armAngle][wristAngle][actionIndex]
						// + explorationBonusValue;
						// maxExplorationBonusValue = explorationBonusValue;
						// Save max action index
						maxAction = actionIndex;
					}
				}
			} catch (Exception ex) {
				ex.printStackTrace();
				// maxActionValue = 0;
			}

			// If randomness is less than a random roll
			if (randomness <= Math.random()) {
				// Pick the max action
				action = maxAction;
				actionValue = maxActionValue;
			} else {
				// Get the value for the random action
				actionValue = QValues[armAngle][wristAngle][action];
			}

			// Get the old value
			oldValue = QValues[previousArmAngle][previousWristAngle][previousAction];

			// Calculate exploration bonus
			// previousExplorationValue =
			// ((float) explorationBonus)
			// / SACounts[previousArmAngle][previousWristAngle][previousAction];
			//
			// if (Float.isInfinite(previousExplorationValue)) {
			// previousExplorationValue = 10;
			// }
			//
			// if (previousExplorationValue < 0.01f) {
			// previousExplorationValue = 0;
			// }

			// Calculate the new QValue
			newValue = (float) (
					((1 - learningRate)
					*
					oldValue
					)
					)
					+
					(
					(learningRate * (
					((float) (reward + (futureDiscount * maxActionValue)))
					// +
					// previousExplorationValue - oldValue
					)));

			// if (Float.isNaN(newValue)) {
			// System.out.println("Error: newValue NaN");
			// newValue = worstValue;
			// }
			// if (Float.isInfinite(newValue)) {
			// System.out.println("Error: newValue is infinite");
			// newValue = bestValue;
			// }

			// newValue = MyMath.lerp(oldValue, newValue, 0.5f);

			// Check if value is best or worst seen so far
			if (newValue > bestValue) {
				bestValue = newValue;
			} else if (newValue < worstValue) {
				worstValue = newValue;
			}

			// If close to best or worst
			if (newValue > bestValue * .5f) {
				// Reset time since good value
				timeSinceGoodValue = 0;

				// Move learning rate toward min
				learningRate = MyMath.lerp(learningRate, minLearningRate, impatience);
				// Move randomness toward min
				randomness = MyMath.lerp(randomness, minRandomness, impatience);
			} else {

				timeSinceGoodValue += delta;
				if (timeSinceGoodValue > 720) {
					randomness = MyMath.lerp(randomness, maxRandomness, impatience
							* timeSinceGoodValue);
					learningRate = MyMath.lerp(learningRate, maxLearningRate, impatience
							* timeSinceGoodValue);
				}

				// If more than ten minutres without good value
				if (timeSinceGoodValue > 720) {
					timeSinceGoodValue = 0;
					// sendHome();
					CrawlingCrate child = (CrawlingCrate) play.population.makeCrawlingCrate();
					child.sendHome();
					// If selected
					if (this == play.population.selectedPlayer) {
						// Change to child
						play.changePlayer(child);
					}
					// Remove this player
					play.population.removePlayer(this);

				}
			}

			// Save the update
			QValues[previousArmAngle][previousWristAngle][previousAction] = newValue;

			// If is not manual control
			if (!isManualControl) {
				// Take the chosen action, act() saves the action taken
				act(action);
				// Else not manual control
			} else {
				// Save the action taken
				// previousAction = action;
			}

			// Save current values and state as next step's previous
			previousSpeed = speed;
			previousReward = reward;
			previousMaxAction = maxAction;
			previousMaxActionValue = maxActionValue;
			previousValue = newValue;
			if (isDebug) {
				previousActions.add(previousAction);
				previousValues.add(((int) newValue));
			}
			previousArmAngle = armAngle;
			previousWristAngle = wristAngle;
			// previousStates.add(new int[] { previousArmAngle, previousWristAngle });

		} catch (Exception ex) {
			ex.printStackTrace();
		}

	}

	public void update(float delta) {
		// Update age
		age = age.plus((long) (delta * 1000));

		// Update timer
		updateTime += delta;

		// If time to update
		if (updateTime > updateTimer) {
			// Reset update timer
			updateTime = 0.0f;

			// Perform Q Update
			QUpdate(delta);
			QFunctionUpdate(delta);

			
			if (body.getPosition().x >= finishLine) {
				// Push finish line back
//				finishLine = finishLine + (int)(finishLine * .25f);
//				finishLine = Math.min(finishLine, 10000);
				play.finishLine(this);
			}
			
			if (body.getPosition().y < -100) {
				sendHome();
			}// If outside range
			if (previousWristAngle <= 0) {
				if (wristJoint.getMotorSpeed() < 0) {
					// System.out.println("Error: wrist angle below 0");
					wristJoint.enableMotor(true);
					wristJoint.setMotorSpeed(wristSpeed * .01f);
					return;
				}
			}
			if (previousWristAngle >= QValues[0].length) {
				if (wristJoint.getMotorSpeed() > 0) {
					// System.out.println("Error: wrist angle above range");
					wristJoint.enableMotor(true);
					wristJoint.setMotorSpeed(-wristSpeed * .01f);
					return;
				}

			}
			if (previousArmAngle <= 0) {
				if (armJoint.getMotorSpeed() < 0) {
					// System.out.println("Error: arm angle below 0");
					armJoint.enableMotor(true);
					armJoint.setMotorSpeed(armSpeed * .01f);
					return;
				}
			}
			if (previousArmAngle >= QValues.length -1) {
				if (armJoint.getMotorSpeed() > 0) {
					// System.out.println("Error: arm angle above range");
					armJoint.enableMotor(true);
					armJoint.setMotorSpeed(-armSpeed * .01f);
					return;
				}
			}
			
		}
		

		
	}
	

	public void render(SpriteBatch spriteBatch, Camera camera,
			float delta, boolean showFinish) {

		// Get the color for the reward display
		Color color;
		float value = fastSig(previousValue);

		if (previousValue > 0) {
			// value = previousValue / bestValue;
			color = new Color(0f, value, 0f, 1);
		} else if (previousValue < 0) {
			// value = previousValue / worstValue;
			color = new Color(value, 0f, 0f, 1);
		} else {
			color = Color.BLACK;
		}
		float[] colors = new float[] { color.r, color.g, color.b };

		// Draw reward
		spriteBatch.setProjectionMatrix(camera.combined);
		spriteBatch.begin();

		bodyTrail.getTint().setColors(colors);
		bodyTrail.getScale().setHigh(value * .5f);
		bodyTrail.setPosition(body.getPosition().x, body.getPosition().y + (height * 2f));
		bodyTrail.draw(spriteBatch, delta);
		
		finishParticleEmitter.setPosition(finishLine, 0);
		finishParticleEmitter.draw(spriteBatch, delta);

		// Show name
		if (showName) {
			font.draw(spriteBatch, rank + " " + name, body.getPosition().x,
					body.getPosition().y + 12);
		}
		spriteBatch.end();

	}

	// getPhysicalMenu() - Get the part of the physical menu pertaining to the crawler
	public Table getPhysicalMenu() {
		int slideWidth = GamePreferences.instance.slideWidth;
		int padding = GamePreferences.instance.padding;

		Table tbl = new Table();
		tbl.columnDefaults(0).padRight(padding);
		tbl.columnDefaults(1).padRight(padding);
		tbl.columnDefaults(2).padRight(padding);

		// Arm Range Slider
		tbl.add(new Label("Arm Range: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		final Slider sldArmRange = new Slider(0, 360, 1, false, Assets.instance.skin);
		sldArmRange.setValue(armRange);
		sldArmRange.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				armRange = (int) ((Slider) actor).getValue();
				armJoint.setLimits(0, (int) Math.toRadians(armRange));
				GamePreferences.instance.armRange = armRange;
			}
		});
		tbl.add(sldArmRange).width(slideWidth);
		tbl.add(new Label("360", Assets.instance.skin));
		tbl.row();

		// Wrist Range Slider
		tbl.add(new Label("Wrist Range: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		final Slider sldWristRange = new Slider(0, 360, 1, false, Assets.instance.skin);
		sldWristRange.setValue(wristRange);
		sldWristRange.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				wristRange = (int) ((Slider) actor).getValue();
				wristJoint.setLimits(0, (int) Math.toRadians(wristRange));
				GamePreferences.instance.wristRange = wristRange;
			}
		});
		tbl.add(sldWristRange).width(slideWidth);
		tbl.add(new Label("360", Assets.instance.skin));
		tbl.row();

		// Suspension Slider
		tbl.add(new Label("Suspension: ", Assets.instance.skin));
		tbl.add(new Label("1", Assets.instance.skin));
		final Slider sldSuspension = new Slider(0, 4, .01f, false, Assets.instance.skin);
		sldSuspension.setValue(leftAxis.getSpringFrequencyHz());
		sldSuspension.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				leftAxis.setSpringFrequencyHz(((Slider) actor).getValue());

				rightAxis.setSpringFrequencyHz(((Slider) actor).getValue());
				GamePreferences.instance.suspension = ((Slider) actor).getValue();
			}
		});
		tbl.add(sldSuspension).width(slideWidth);
		tbl.add(new Label("4", Assets.instance.skin));
		tbl.row();

		// Friction Slider
		tbl.add(new Label("Friction: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		final Slider sldFriction = new Slider(0, 1000, .01f, false, Assets.instance.skin);
		sldFriction.setValue(wrist.getFixtureList().first().getFriction());
		sldFriction.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = ((Slider) actor).getValue();
				GamePreferences.instance.friction = value;
				wrist.getFixtureList().first().setFriction(value);
			}
		});
		tbl.add(sldFriction).width(slideWidth);
		tbl.add(new Label("1000", Assets.instance.skin));
		tbl.row();

		// Density Slider
		tbl.add(new Label("Density: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		final Slider slider = new Slider(0, 10, .01f, false, Assets.instance.skin);
		slider.setValue(density);
		slider.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = ((Slider) actor).getValue();
				GamePreferences.instance.density = value;

				body.getFixtureList().get(0).setDensity(value);
			}
		});
		tbl.add(slider).width(slideWidth);
		tbl.add(new Label("10", Assets.instance.skin));
		tbl.row();

		// Dampening Slider
		tbl.add(new Label("Dampening: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		final Slider sldDampening = new Slider(0, 2, .01f, false, Assets.instance.skin);
		sldDampening.setValue(body.getLinearDamping());
		sldDampening.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				GamePreferences.instance.linearDampening = ((Slider) actor).getValue();
				body.setLinearDamping(((Slider) actor).getValue());
			}
		});
		tbl.add(sldDampening).width(slideWidth);
		tbl.add(new Label("10", Assets.instance.skin));
		tbl.row();

		return tbl;
	}

	// getLearningMenu() - Get the part of the learning menu pertaining to the crawler
	@Override
	public Table getLearningMenu() {
		int padding = GamePreferences.instance.padding;
		int slideWidth = GamePreferences.instance.slideWidth;

		Table tbl = new Table();

		// Speed Value Weight Slider
		tbl.add(new Label("Speed Value: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		final Slider sldSpeedValue = new Slider(0, 10, .01f, false, Assets.instance.skin);
		sldSpeedValue.setValue(speedValueWeight);
		sldSpeedValue.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = ((Slider) actor).getValue();
				speedValueWeight = value;
				GamePreferences.instance.speedValueWeight = value;
			}
		});
		tbl.add(sldSpeedValue).width(slideWidth);
		tbl.add(new Label("10", Assets.instance.skin));
		tbl.row();

		// Acceleration Value Weight Slider
		tbl.add(new Label("Acceleration Value: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		final Slider sldAccelerationValue = new Slider(0, 10, .01f, false, Assets.instance.skin);
		sldAccelerationValue.setValue(averageSpeedValueWeight);
		sldAccelerationValue.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = ((Slider) actor).getValue();
				averageSpeedValueWeight = value;
			}
		});
		tbl.add(sldAccelerationValue).width(slideWidth);
		tbl.add(new Label("10", Assets.instance.skin));
		tbl.row();
		tbl.pack();

		return tbl;
	}

	// TODO: Finish this method
	public Table getStatsWindow() {
		int padding = 10;

		int slideWidth = 100;

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

		return tbl;
	}

	// LearnFromLeader(Player) - Move the Q-Values toward Player's Q-Values, with learning rate
	public void learnFromLeader(Player leader, float learningRate) {

		if (this.equals(leader) || this == leader) {
			return;
		}

		// Transfer QTable
		for (int x = 0; x < QValues.length; x++) {
			for (int y = 0; y < QValues[x].length; y++) {
				for (int z = 0; z < QValues[x][y].length; z++) {
					// update Q-Value, reduced by learning rate
					QValues[x][y][z] = ((1 - learningRate) * QValues[x][y][z])
							+ (learningRate * ((CrawlingCrate) leader).QValues[x][y][z]);
					// throw in a bit of randomness
					// QValues[x][y][z] += (Math.random() > 0.5f ? -0.001 : 0.001);

					// SACounts[x][y][z] += (((CrawlingCrate) leader)).SACounts[x][y][z];
				}
			}
		}
	}

	public void learnFromAll(ArrayList<BasicPlayer> allAgents, float learningRate) {

		for (BasicPlayer agent : allAgents) {

			if (agent.equals(this) || this == agent) {
				continue;
			}
			if (agent instanceof BasicAgent) {
				if (agent instanceof CrawlingCrate) {

					// Transfer QTable
					for (int x = 0; x < QValues.length; x++) {
						for (int y = 0; y < QValues[x].length; y++) {
							for (int z = 0; z < QValues[x][y].length; z++) {
								// update Q-Value, reduced by learning rate
								QValues[x][y][z] = ((1 - learningRate) * QValues[x][y][z])
										+ (learningRate * ((CrawlingCrate) agent).QValues[x][y][z]);
								// throw in a bit of randomness
								// QValues[x][y][z] += (Math.random() > 0.5f ? -0.001 : 0.001);
								// SACounts[x][y][z] += (((CrawlingCrate) agent)).SACounts[x][y][z];
							}
						}
					}
				}
			}
		}
	}

	public CrawlingCrate spawn(World world) {
		CrawlingCrate child = new CrawlingCrate(this.play);
		Vector2 position = body.getPosition();

		child.isDebug = false;
		float mutatedWidth = width
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f) * (mutationRate * 2.0 * width * Math
						.random()));
		float mutatedHeight = height
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f) * (mutationRate * 2.0 * height * Math
						.random()));

		float mutatedUpdateTimer = updateTimer
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f) * (mutationRate * updateTimer * Math
						.random()));
		float mutatedDensity = density
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f) * (mutationRate * density * Math
						.random()));
		float mutatedMinRandomness = minRandomness
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f) * (mutationRate * minRandomness * Math.random()));
		float mutatedMaxRandomness = maxRandomness
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f) * (mutationRate * maxRandomness * Math
						.random()));
		float mutatedMinLearningRate = minLearningRate
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f) * (mutationRate * minLearningRate * Math
						.random()));
		float mutatedMaxLearningRate = maxLearningRate
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f) * (mutationRate * maxLearningRate * Math
						.random()));
		float mutatedArmSpeed = armSpeed
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f) * (mutationRate * armSpeed * Math
						.random()));
		float mutatedWristSpeed = wristSpeed
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f) * (mutationRate * wristSpeed * Math
						.random()));
		float mutatedArmTorque = armTorque
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f) * (mutationRate * armTorque * Math
						.random()));
		float mutatedWristTorque = wristTorque
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f) * (mutationRate * wristTorque * Math
						.random()));
		float mutatedFutureDiscount = futureDiscount
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f)
				* (mutationRate * futureDiscount * Math.random()));
		float mutatedSpeedWeight = speedValueWeight
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 0.5 * speedValueWeight * Math.random()));
		float mutatedAverageSpeedWeight = averageSpeedValueWeight
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 0.5 * averageSpeedValueWeight * Math.random()));

		float mutatedLegSpread = legSpread
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 2.0 * legSpread * Math.random()));

		float mutatedWheelRadius = wheelRadius
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 1.5 * wheelRadius * Math.random()));

		float mutatedSuspension = suspension
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 0.5 * suspension * Math.random()));
		
		float mutatedArmLength = armLength
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 2.0 * armLength * Math.random()));

		float mutatedArmWidth = armWidth
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 2.0 * armWidth * Math.random()));
		
		float mutatedWristLength = wristLength
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 2.0 * armLength * Math.random()));
		
		float mutatedWristWidth = wristWidth
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 2.0 * wristWidth * Math.random()));

		child.init(world,
				position.x, position.y,
				mutatedWidth, mutatedHeight,
				mutatedMaxRandomness, mutatedMaxLearningRate,
				mutatedMinRandomness, mutatedMaxRandomness,
				mutatedMinLearningRate, mutatedMaxLearningRate,
				learningRateDecay, mutatedUpdateTimer,
				mutatedFutureDiscount, mutatedSpeedWeight,
				mutatedAverageSpeedWeight,
				mutatedArmSpeed, mutatedWristSpeed,
				armRange, wristRange,
				mutatedArmTorque, mutatedWristTorque, bodyShape,
				mutatedDensity, mutatedLegSpread, mutatedWheelRadius, mutatedSuspension,
				rideHeight,
				mutatedArmLength, mutatedArmWidth,mutatedWristLength, mutatedWristWidth);

		return child;
	}

	public CrawlingCrate clone(World world) {
		CrawlingCrate child = new CrawlingCrate(this.play);
		Vector2 position = body.getPosition();

		// child.isDebug = true;

		child.init(world, position.x, position.y, width, height, maxRandomness, maxLearningRate,
				minRandomness, maxRandomness,
				minLearningRate, maxLearningRate, learningRateDecay, updateTimer, futureDiscount,
				speedValueWeight,
				averageSpeedValueWeight,
				armSpeed, wristSpeed, armRange, wristRange, armTorque, wristTorque, bodyShape,
				density);

		child.learnFromLeader(this, 1);

		return child;
	}

	@Override
	public boolean keyDown(int keycode) {
		// System.out.println("keyDown()");
		switch (keycode) {
			case Keys.UP:
				act(2);
				// armJoint.enableMotor(true);
				// armJoint.setMotorSpeed(armSpeed);
				break;
			case Keys.DOWN:
				act(3);
				// armJoint.enableMotor(true);
				// armJoint.setMotorSpeed(-armSpeed);
				break;
			case Keys.RIGHT:
				act(0);
				// wristJoint.enableMotor(true);
				// wristJoint.setMotorSpeed(wristSpeed);
				break;
			case Keys.LEFT:
				act(1);
				// wristJoint.enableMotor(true);
				// wristJoint.setMotorSpeed(-wristSpeed);
				break;

			case Keys.L:
				isManualControl = !isManualControl;
				break;
		}
		return super.keyDown(keycode);
	}

	@Override
	public boolean keyUp(int keycode) {
		// System.out.println("keyUp()");
		switch (keycode) {
			case Keys.DOWN:
				if (armJoint.getMotorSpeed() > 0) {
					armJoint.enableMotor(holdMotors);
					armJoint.setMotorSpeed(0);
				}
				break;
			case Keys.UP:
				if (armJoint.getMotorSpeed() < 0) {
					armJoint.enableMotor(holdMotors);
					armJoint.setMotorSpeed(0);
				}
				break;
			case Keys.RIGHT:
				if (wristJoint.getMotorSpeed() > 0) {
					wristJoint.enableMotor(holdMotors);
					wristJoint.setMotorSpeed(0);
				}
				break;
			case Keys.LEFT:
				if (wristJoint.getMotorSpeed() < 0) {
					wristJoint.enableMotor(holdMotors);
					wristJoint.setMotorSpeed(0);
				}
				break;

		}
		return super.keyUp(keycode);
	}

	@Override
	public void sendHome() {
		bestValue = 0;
		worstValue = 0;

		// Reset visit counts
		// SACounts = new int[armRange][wristRange][actionCount];
		//
		// for (int armIndex = 0; armIndex < armRange; armIndex++) {
		// for (int wristIndex = 0; wristIndex < wristRange; wristIndex++) {
		// for (int actionIndex = 0; actionIndex < actionCount; actionIndex++) {
		// SACounts[armIndex][wristIndex][actionIndex] = 0;
		// }
		// }
		// }
		super.sendHome();
	};

	/* Gettets and Setters */
	@Override
	public ArrayList<Body> getBodies() {
		ArrayList<Body> bodies = new ArrayList<Body>();
		bodies.add(body);
		// bodies.add(leftArm);
		// bodies.add(leftFoot);
		// bodies.add(leftWrist);
		bodies.add(arm);
		// bodies.add(rightFoot);
		bodies.add(wrist);
		bodies.add(leftWheel);
		bodies.add(rightWheel);
		return bodies;
	}

	public ArrayList<String> getStats() {
		ArrayList<String> stats = new ArrayList<String>();
		// stats.add("Goal:" + goals[goal]);
//		stats.add("Name:" + name);
//		stats.add("Rank:" + rank);
//		stats.add("Finish:" + finishLine);
		stats.add("X:" + StringHelper.getDecimalFormat(body.getPosition().x, 0));
		if (age.getStandardHours() > 0) {
			stats.add("Age:" + age.getStandardHours() + "h");
		} else if (age.getStandardMinutes() > 0) {
			stats.add("Age:" + age.getStandardMinutes() + "m");
		} else {
			stats.add("Age:" + age.getStandardSeconds() + "s");
		}
//		stats.add("Mutation Rate:" + mutationRate);

		// stats.add("Reward:" + String.format("%+1.6f", previousReward));
		// stats.add("Exploration Bonus:" + String.format("%+1.6f", previousExplorationValue));
		// stats.add("QValue:" + String.format("%+1.6f", previousValue));

		stats.add("Speed:" + String.format("%+1.2f", speed));
		stats.add("Acceleration:" + String.format("%+1.2f", acceleration));
		stats.add("Reward:" + String.format("%+1.2f", previousReward));
		// stats.add("Arm Angle:" + StringHelper.getDecimalFormat(previousArmAngle, 1));
		// stats.add("Wrist Angle:" + StringHelper.getDecimalFormat(previousWristAngle, 1));
		stats.add("Angles:" + StringHelper.getDecimalFormat(previousArmAngle, 1) + ", "
				+ StringHelper.getDecimalFormat(previousWristAngle, 1));
//		stats.add("State = " + Arrays.toString(state));

		// stats.add("Best Value:" + String.format("%+1.6f", bestValue));
		// stats.add("Worst Value:" + String.format("%+1.6f", worstValue));
		stats.add("Max Speed:" + String.format("%+1.2f", maxSpeed));
		stats.add("Value Range:" + String.format("%+1.2f", worstValue) + ", "
				+ String.format("%+1.2f", bestValue));
//		stats.add("TimeSinceGoodValue:" + String.format("%+1.6f", timeSinceGoodValue));
		 stats.add("Impatience:" + String.format("%+1.6f", impatience));
		 stats.add("Learning Rate:" + String.format("%+1.6f", learningRate));
		 stats.add("Randomness:" + String.format("%+1.6f", randomness));

		// stats.add("MaxAction:" + previousMaxAction);
		// stats.add("MaxActionValue:" + String.format("%+1.6f", previousMaxActionValue));
		// stats.add("    Discounted:"
		// + String.format("%+1.6f", previousMaxActionValue * futureDiscount));

		// stats.add("Action:" + previousAction
		// + ((previousAction != previousMaxAction) ? "*" : ""));

		// stats.add("Visited " + SACounts[previousArmAngle][previousWristAngle][previousAction]);
		stats.add("Actions:" + previousActions.toString());
		// stats.add("QValues:" + previousValues.toString());
		// stats.add("Old QValue:" + String.format("%+1.6f", oldValue));
		// stats.add("New QValue:" + String.format("%+1.6f", newValue));
		// stats.add("QValue Delta:" + String.format("%+1.6f", newValue - oldValue));

		stats.add("QValue:" + String.format("%+1.2f", oldValue) + "->"
				+ String.format("%+1.2f", oldValue) + " ("
				+ String.format("%+1.2f", newValue - oldValue) + ")");
		// stats.add("UpdateTimer:" + String.format("%+1.6f", updateTimer));

//		stats.add("Q-Values: ["
//				+ String.format("%+1.2f", QValues[previousArmAngle][previousWristAngle][0])
//				+ ", " + String.format("%+1.2f", QValues[previousArmAngle][previousWristAngle][1])
//				+ ", "
//				+ String.format("%+1.2f", QValues[previousArmAngle][previousWristAngle][2]) + ", "
//				+ String.format("%+1.2f", QValues[previousArmAngle][previousWristAngle][3]) + ", "
//				+ String.format("%+1.2f", QValues[previousArmAngle][previousWristAngle][4]) + ", "
//				+ String.format("%+1.2f", QValues[previousArmAngle][previousWristAngle][5]) + " ] ");
//		stats.add("Action Visits: [" + SACounts[previousArmAngle][previousWristAngle][0]
//				+ ", " + SACounts[previousArmAngle][previousWristAngle][1] + ", "
//				+ SACounts[previousArmAngle][previousWristAngle][2] + ", "
//				+ SACounts[previousArmAngle][previousWristAngle][3] + ", "
//				+ SACounts[previousArmAngle][previousWristAngle][4] + ", "
//				+ SACounts[previousArmAngle][previousWristAngle][5] + " ] ");
		return stats;

	}

	public Body getBody() {
		return body;
	}

	// Save Q-State
	public void saveState() {
		try {
			// Gson gson = new Gson();
			String content = "This is the content to write into file";

			File file = new File("/users/mkyong/filename.txt");

			// if file doesnt exists, then create it
			if (!file.exists()) {
				file.createNewFile();
			}

			FileWriter fw = new FileWriter(file.getAbsoluteFile());
			BufferedWriter bw = new BufferedWriter(fw);
			bw.write(content);
			bw.close();

			System.out.println("Done");

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public float getArmSpeed() {
		return armSpeed;
	}

	public void setArmSpeed(float armSpeed) {
		this.armSpeed = armSpeed;
		armJoint.setMotorSpeed(armSpeed);
	}

	public float getWristSpeed() {
		return wristSpeed;
	}

	public void setWristSpeed(float wristSpeed) {
		this.wristSpeed = wristSpeed;
		wristJoint.setMotorSpeed(wristSpeed);
	}

	public int getArmRange() {
		return armRange;
	}

	public void setArmRange(int armRange) {
		this.armRange = armRange;
		armJoint.setLimits(0, (float) Math.toRadians(armRange));
		GamePreferences.instance.armRange = armRange;
	}

	public int getWristRange() {
		return wristRange;
	}

	public void setWristRange(int wristRange) {
		this.wristRange = wristRange;
		wristJoint.setLimits(0, (float) Math.toRadians(wristRange));
		GamePreferences.instance.wristRange = wristRange;

	}

	public float getArmTorque() {
		return armTorque;
	}

	public void setArmTorque(float armTorque) {
		this.armTorque = armTorque;
		this.armJoint.setMaxMotorTorque(armTorque);
	}

	public float getWristTorque() {
		return wristTorque;
	}

	public void setWristTorque(float wristTorque) {
		this.wristTorque = wristTorque;
		this.wristJoint.setMaxMotorTorque(wristTorque);
	}

	public float getSpeedValueWeight() {
		return speedValueWeight;
	}

	public void setSpeedValueWeight(float speedValueWeight) {
		this.speedValueWeight = speedValueWeight;
	}

	public float getAverageSpeedValueWeight() {
		return averageSpeedValueWeight;
	}

	public void setAverageSpeedValueWeight(float averageSpeedValueWeight) {
		this.averageSpeedValueWeight = averageSpeedValueWeight;
	}

}