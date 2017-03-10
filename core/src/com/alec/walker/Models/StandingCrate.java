package com.alec.walker.Models;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.joda.time.Duration;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.iter.NdIndexIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

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
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.ParticleEmitter;
import com.badlogic.gdx.graphics.g2d.Sprite;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer.ShapeType;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.physics.box2d.Body;
import com.badlogic.gdx.physics.box2d.BodyDef;
import com.badlogic.gdx.physics.box2d.BodyDef.BodyType;
import com.badlogic.gdx.physics.box2d.CircleShape;
import com.badlogic.gdx.physics.box2d.FixtureDef;
import com.badlogic.gdx.physics.box2d.Joint;
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
import com.google.common.collect.EvictingQueue;

//import org.nd4j.linalg.activations.Activation;

public class StandingCrate extends BasicAgent {
	public static final String		TAG					= StandingCrate.class.getName();

	// Keep a reference to play for convenience
	public Play						play;

	public ArrayList<Body>			bodies;
	public ArrayList<Joint>			joints;
	// Body parts
	public Body						body, rightArm, rightWrist,
									leftWheel, rightWheel, middleWheel;
	public Body						leftArm, leftWrist;
	public Body						leftFoot;
	public Body						rightFoot;

	// Joints: arm and wheels
	public RevoluteJoint			rightArmJoint;
	public RevoluteJoint			rightWristJoint;
	public RevoluteJoint			rightFootJoint;
	public RevoluteJoint			leftArmJoint;
	public RevoluteJoint			leftWristJoint;
	public RevoluteJoint			leftFootJoint;
	public WheelJoint				leftAxis, rightAxis, middleAxis;

	// Particle emitter for visualizing reward signal
	public ParticleEmitter			bodyTrail;
	public ParticleEmitter			finishParticleEmitter;
	public BitmapFont				font;
	// Size
	public float					width, height;
	public float[]					bodyShape;
	public float					density;

	// Position
	// Arm parameters
	public float					armWidth;
	public float					wristWidth;
	public float					footWidth;
	public float					armLength;
	public float					wristLength;
	public float					armSpeed;
	public float					wristSpeed;
	public int						armRange;
	public int						wristRange;
	public float					armTorque;
	public float					wristTorque;

	// Body Properties
	public float					legSpread;
	public float					wheelRadius;
	public float					suspension;
	public float					rideHeight;

	// Performance stats
	public float					speed;
	public float					maxSpeed;
	public float					bestSpeed			= 0.5f;
	public float					worstSpeed			= -0.5f;
	public float					speedDecay;
	public float					previousSpeed;
	public float					acceleration;

	// Q-Values
	public int						stateFeatureCount;
	public int						previousArmAngle, previousWristAngle;
	public float					previousReward, previousExplorationValue;
	public float					previousExpectedReward;
	public float					oldValue, newValue;
	public float					timeSinceGoodValue;
	public float					qDifference;

	public float					valueScore			= 0;

	// Race
	public int						rank;
	public int						finishLine			= 100;

	public boolean					isPastFinish		= false;
	public boolean					isUsingEnergy		= false;

	public int						isTouchingGround	= 0;

	// State is each state property
	public float					speedValueWeight, averageSpeedValueWeight;

	public float					exploreBonus;
	// Control the reduction in state space
	public boolean					holdMotors			= true;
	public boolean					showName			= false;

	public MultiLayerNetwork		actorBrain;
	public MultiLayerNetwork		criticBrain;

	// public INDArray expectedRewards;
	public File						locationToSave		= new File("CrawlingCrate_Brain.zip");	// Where to save the network. Note: the file is in
																								// .zip format - can be opened
	// externally
	public float[]					previousState;
	private INDArray				previousStateNd;
	private INDArray				expectedRewards;
	private INDArray				previousExpectedRewards;

	private EvictingQueue<INDArray>	currentActorTarget;
	private EvictingQueue<INDArray>	currentStates;
	private EvictingQueue<INDArray>	previousStates;
	private EvictingQueue<Float>	currentRewards;
	private EvictingQueue<Integer>	previousActions;

	// Random number generator seed, for reproducability
	public int						memoryCount			= 15;
	public int						actionCount			= 4;
	public int						weightCount			= 4;
	public static final int			seed				= 12345;
	public static final int			iterations			= 1;
	public static final double		learningRateNN		= 0.001f;
	public static final Random		rng					= new Random(seed);

	public long[]					actionVisitCounts;
	public long						updateCount;

	public StandingCrate(Play play) {
		this.play = play;

		rank = 0;
		currentStates = EvictingQueue.create(memoryCount);
		previousStates = EvictingQueue.create(memoryCount);
		currentRewards = EvictingQueue.create(memoryCount);
		previousActions = EvictingQueue.create(memoryCount);
		currentActorTarget = EvictingQueue.create(memoryCount);

	}

	public void init(World world, float x, float y) {
		float defaultWidth = 5;
		float defaultHeight = 9;
		init(world, x, y, defaultWidth, defaultHeight);
	}

	public void init(World world, float x, float y,
			float width, float height) {

		speedValueWeight = GamePreferences.instance.speedValueWeight;
		averageSpeedValueWeight = GamePreferences.instance.speedValueWeight * 0.5f;

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
		float defaultLegSpread = width * 12;
		float defaultWheelRadius = height / 2f;
		float defaultSuspension = GamePreferences.instance.suspension;
		float rideHeight = height * 0.2f;
		float armWidth = .15f * width;
		float armLength = .3f * height;
		float wristLength = .6f * height;
		float wristWidth = armWidth;
		float footWidth = armWidth;
		// Calculate arm lengths based on body size

		init(world, x, y, width, height, randomness, qlearningRate, minRandomness, maxRandomness,
				minlearningRate, maxLearningRate, learningRateDecay, updateTimer, futureDiscount,
				speedValueWeight, averageSpeedValueWeight, armSpeed, wristSpeed, armRange,
				wristRange, armTorque, wristTorque, bodyShape, density, defaultLegSpread,
				defaultWheelRadius, defaultSuspension, rideHeight, armLength, armWidth,
				wristLength, wristWidth, footWidth);
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
			float wristLength, float wristWidth,
			float footWidth

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
		// No limit
		// this.armRange = 360;
		// this.wristRange = 360;
		this.wristSpeed = wristSpeed;
		this.armTorque = armTorque;
		this.wristRange = wristRange;
		this.wristTorque = wristTorque;
		this.rideHeight = rideHeight;
		this.density = density;
		this.updateTimer = updateTimer;
		this.suspension = suspension;
		this.legSpread = legSpread;
		this.wheelRadius = wheelRadius;
		this.impatience = GamePreferences.instance.impatience;
		this.footWidth = footWidth;

		// Calculate body shape
		float halfWidth = width * 0.5f;
		float halfHeight = height * 0.5f;

		bodyShape = new float[] { -(halfWidth * .95f), -halfHeight, // bottom left
				halfWidth, -halfHeight, // bottom right
				halfWidth + .2f, 0,
				halfWidth * 1.5f, halfHeight * 1.1f,	// top right
				-halfWidth - .2f, halfHeight,	// top left
				-(halfWidth * .95f), -halfHeight // bottom left
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

		initLearning();

		// Define body properties
		BodyDef bodyDef = new BodyDef();
		bodyDef.type = BodyType.DynamicBody;
		bodyDef.position.set(x, y);
		bodyDef.linearDamping = GamePreferences.instance.linearDampening;
		bodyDef.allowSleep = false;
		bodyDef.bullet = false;

		FixtureDef fixtureDef = new FixtureDef();
		fixtureDef.density = density * .75f;
		fixtureDef.friction = .01f;
		fixtureDef.restitution = .01f; // Bounciness
		fixtureDef.filter.categoryBits = Constants.FILTER_CAR; // Interacts as
		fixtureDef.filter.maskBits = Constants.FILTER_BOUNDARY; // Interacts with

		// create the chassis
		PolygonShape chassisShape = new PolygonShape();

		// Set the chassis shape
		chassisShape.set(bodyShape);
		fixtureDef.shape = chassisShape;

		// Create the chassis body
		body = world.createBody(bodyDef);
		body.createFixture(fixtureDef);

		// Turn off linear dampening for rest of body
		bodyDef.linearDamping = 0.0f;

		// Create the arm
		PolygonShape armShape = new PolygonShape();
		armShape.setAsBox(armLength, armWidth);
		fixtureDef.shape = armShape;
		fixtureDef.restitution = .00f; // Bounciness
		fixtureDef.density = density * .35f; // Reduce density of arm
		fixtureDef.filter.categoryBits = Constants.FILTER_CAR; // Interacts as
		fixtureDef.filter.maskBits = Constants.FILTER_BOUNDARY; // Interacts with
		rightArm = world.createBody(bodyDef);
		rightArm.createFixture(fixtureDef);

		leftArm = world.createBody(bodyDef);
		leftArm.createFixture(fixtureDef);

		// arm.setTransform(arm.getPosition(), (float) Math.toRadians(armRange * .5));

		RevoluteJointDef armJointDef = new RevoluteJointDef();
		armJointDef.bodyA = body;
		armJointDef.bodyB = rightArm;
		armJointDef.collideConnected = false;
		armJointDef.localAnchorA.set(new Vector2(0, height * .1f));
		armJointDef.localAnchorB.set(new Vector2((-armLength), 0));
		armJointDef.enableLimit = true;
		armJointDef.maxMotorTorque = armTorque;
		armJointDef.referenceAngle = (float) Math.toRadians(-180);
		armJointDef.lowerAngle = 0;
		armJointDef.upperAngle = (float) Math.toRadians(armRange);
		rightArmJoint = (RevoluteJoint) world.createJoint(armJointDef);
		rightArmJoint.enableMotor(true);

		armJointDef.bodyB = leftArm;
		armJointDef.localAnchorA.set(new Vector2(0, height * .1f));
		armJointDef.localAnchorB.set(new Vector2((-armLength), 0));
		leftArmJoint = (RevoluteJoint) world.createJoint(armJointDef);
		leftArmJoint.enableMotor(true);

		PolygonShape wristShape = new PolygonShape();

		wristShape.set(
				new float[] { -wristWidth, 0, // bottom left
						wristWidth, 0, // bottom right
						wristWidth, wristLength
				});
		// wristShape.setAsBox(wristLength, .05f * height);
		fixtureDef.shape = wristShape;
		fixtureDef.friction = GamePreferences.instance.friction;
		fixtureDef.restitution = 0f;

		rightWrist = world.createBody(bodyDef);
		rightWrist.createFixture(fixtureDef);

		leftWrist = world.createBody(bodyDef);
		leftWrist.createFixture(fixtureDef);

		RevoluteJointDef wristJointDef = new RevoluteJointDef();
		wristJointDef.bodyA = rightArm;
		wristJointDef.bodyB = rightWrist;
		wristJointDef.collideConnected = false;
		wristJointDef.localAnchorA.set(new Vector2(armLength, 0));
		// rightWristJointDef.localAnchorB.set(new Vector2(-wristLength * 1.1f, 0));
		wristJointDef.enableLimit = true;
		wristJointDef.maxMotorTorque = wristTorque;
		wristJointDef.referenceAngle = (float) Math.toRadians(180);
		wristJointDef.lowerAngle = 0;
		wristJointDef.upperAngle = (float) Math.toRadians(wristRange);
		rightWristJoint = (RevoluteJoint) world.createJoint(wristJointDef);
		rightWristJoint.enableMotor(true);

		wristJointDef.bodyA = leftArm;
		wristJointDef.bodyB = leftWrist;
		// wristJointDef.localAnchorA.set(new Vector2(armLength, 0));
		// rightWristJointDef.localAnchorB.set(new Vector2(-wristLength * 1.1f, 0));
		leftWristJoint = (RevoluteJoint) world.createJoint(wristJointDef);
		leftWristJoint.enableMotor(true);

		// Create the feet
		CircleShape footShape = new CircleShape();
		footShape.setRadius(footWidth);
		fixtureDef.shape = footShape;
		fixtureDef.restitution = .00f; // Bounciness
		fixtureDef.friction = .15f;
		fixtureDef.density = density * .25f; // Reduce density of arm
		fixtureDef.filter.categoryBits = Constants.FILTER_CAR; // Interacts as
		fixtureDef.filter.maskBits = Constants.FILTER_BOUNDARY; // Interacts with
		// rightFoot = world.createBody(bodyDef);
		// rightFoot.createFixture(fixtureDef);
		//
		// leftFoot = world.createBody(bodyDef);
		// leftFoot.createFixture(fixtureDef);

		// Create the foot joints
		// RevoluteJointDef footJointDef = new RevoluteJointDef();
		// footJointDef.bodyA = rightWrist;
		// footJointDef.bodyB = rightFoot;
		// footJointDef.collideConnected = false;
		// footJointDef.localAnchorA.set(new Vector2(footWidth, wristLength));
		// // rightWristJointDef.localAnchorB.set(new Vector2(-wristLength * 1.1f, 0));
		// footJointDef.enableLimit = true;
		// footJointDef.maxMotorTorque = wristTorque;
		// footJointDef.referenceAngle = (float) Math.toRadians(110);
		// footJointDef.lowerAngle = 0;
		// footJointDef.upperAngle = (float) Math.toRadians(wristRange);
		// rightFootJoint = (RevoluteJoint) world.createJoint(footJointDef);
		// rightFootJoint.enableMotor(true);

		// footJointDef.bodyA = leftWrist;
		// footJointDef.bodyB = leftFoot;
		// footJointDef.localAnchorA.set(new Vector2(footWidth, wristLength));
		// // rightWristJointDef.localAnchorB.set(new Vector2(-wristLength * 1.1f, 0));
		// leftFootJoint = (RevoluteJoint) world.createJoint(footJointDef);
		// leftFootJoint.enableMotor(true);

		// create the wheels
		CircleShape wheelShape = new CircleShape();
		wheelShape.setRadius(wheelRadius * 0.25f);
		Sprite wheelSprite = new Sprite(new Texture("data/img/car/wheel.png"));
		wheelSprite.setSize(wheelShape.getRadius(), wheelShape.getRadius());
		wheelSprite.setOrigin(wheelSprite.getWidth() / 2, wheelSprite.getHeight() / 2);

		// Wheel fixture def
		FixtureDef wheelFixtureDef = new FixtureDef();
		wheelFixtureDef.density = density * 4f;
		wheelFixtureDef.friction = .001f;
		wheelFixtureDef.restitution = 0;
		wheelFixtureDef.filter.categoryBits = Constants.FILTER_CAR; // Interacts as
		wheelFixtureDef.filter.maskBits = Constants.FILTER_BOUNDARY; // Interacts with
		wheelFixtureDef.shape = wheelShape;

		leftWheel = world.createBody(bodyDef);
		leftWheel.createFixture(wheelFixtureDef);
		// leftWheel.setUserData(wheelSprite);

		// middleWheel = world.createBody(bodyDef);
		// middleWheel.createFixture(wheelFixtureDef);
		// middleWheel.setUserData(wheelSprite);

		rightWheel = world.createBody(bodyDef);
		rightWheel.createFixture(wheelFixtureDef);
		// rightWheel.setUserData(wheelSprite);

		// create the axis'
		WheelJointDef axisDef = new WheelJointDef();
		axisDef.bodyA = body;
		axisDef.localAxisA.set(Vector2.Y);
		axisDef.frequencyHz = suspension;
		axisDef.maxMotorTorque = 15.0f;

		axisDef.bodyB = leftWheel;
		axisDef.localAnchorA.set(-(width) + wheelShape.getRadius() * 0.5f,
				-(halfHeight) - wheelShape.getRadius() * 2 - rideHeight);
		// axisDef.localAnchorB.set(0,-rideHeight);

		leftAxis = (WheelJoint) world.createJoint(axisDef);
		leftAxis.enableMotor(false);
		leftAxis.setSpringFrequencyHz(suspension);

		axisDef.bodyB = rightWheel;
		axisDef.localAnchorA.x *= -1;
		rightAxis = (WheelJoint) world.createJoint(axisDef);
		rightAxis.enableMotor(false);
		rightAxis.setSpringFrequencyHz(suspension);

		axisDef.bodyB = middleWheel;
		axisDef.localAnchorA.x = 0;
		axisDef.localAnchorA.y += 0.1f;
		// axisDef.localAnchorA.y = -height - (rideHeight * .85f);
		// middleAxis = (WheelJoint) world.createJoint(axisDef);
		// middleAxis.enableMotor(false);
		// middleAxis.setSpringFrequencyHz(suspension);

		// Particle Emitters
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
		// finishParticleEmitter.getScale().setHigh(.5f);

		finishParticleEmitter.getTint().setActive(true);
		finishParticleEmitter.getGravity().setActive(false);
		finishParticleEmitter.getVelocity().setActive(false);
		// finishParticleEmitter.getTint().setColors(new float[] { 0.0f, 1f, 0f });
		finishParticleEmitter.start();

		// Set body's user data to this so it can be referenced from game
		body.setUserData(this);
		leftWrist.setUserData(this);
		rightWrist.setUserData(this);

		bodies = new ArrayList<Body>();
		joints = new ArrayList<Joint>();

		// Keep all bodies in a list
		bodies.add(body);
		bodies.add(rightArm);
		bodies.add(rightWrist);
		bodies.add(leftArm);
		bodies.add(leftWrist);
		bodies.add(leftWheel);
		// bodies.add(middleWheel);
		bodies.add(rightWheel);
		// bodies.add(rightFoot);
		// bodies.add(leftFoot);

		// bodies.add(middleAxis);
		joints.add(rightAxis);
		joints.add(leftAxis);
		joints.add(leftArmJoint);
		joints.add(rightArmJoint);
		joints.add(leftWristJoint);
		joints.add(rightWristJoint);
		joints.add(leftFootJoint);
		joints.add(rightFootJoint);
	}

	public void initLearning() {
		// System.out.println("initLearning()");
		age = new Duration(0);

		previousAction = 0;
		previousVelocity = 0;
		bestValue = 0;
		worstValue = 0;
		valueDelta = 0;
		valueVelocity = 0;
		timeSinceGoodValue = 0;
		exploreBonus = 0;

		actionVisitCounts = new long[actionCount];
		for (int index = 0; index < actionCount; index++) {
			actionVisitCounts[index] = 1;
		}
		updateCount = 1;

		// int numInput = weightCount;
		int numInput = weightCount;
		int numOutputs = actionCount;
		// int numOutputs = weightCount * actionCount + 1;
		int nHidden = 16;
		// Create the network
		actorBrain = new MultiLayerNetwork(

				new NeuralNetConfiguration.Builder()
						.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
						.updater(Updater.ADAM)
						.momentum(0.9)
						.learningRate(learningRateNN)
						.iterations(iterations)
						.list(
								new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
										.activation(Activation.ELU).build(),
								new DenseLayer.Builder().nIn(nHidden).nOut(nHidden)
										.activation(Activation.ELU).build(),
								new DenseLayer.Builder().nIn(nHidden).nOut(nHidden)
										.activation(Activation.ELU).build(),
								new DenseLayer.Builder().nIn(nHidden).nOut(nHidden)
										.activation(Activation.ELU).build(),
								new OutputLayer.Builder(LossFunction.MSE)
										.activation(Activation.IDENTITY).nIn(nHidden)
										.nOut(numOutputs).build()
						).pretrain(false).backprop(true).build()
				);

		actorBrain.init();
		// Create the critic network
		criticBrain = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.learningRate(learningRateNN)
				.weightInit(WeightInit.XAVIER)
				.updater(Updater.ADAM)
				.list()
				.layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
						.activation("relu")
						// .dropOut(0.1)
						.build())

				.layer(1, new DenseLayer.Builder().nIn(nHidden).nOut(nHidden)
						.activation("relu")
						// .dropOut(0.2)
						.build())
				.layer(2, new DenseLayer.Builder().nIn(nHidden).nOut(nHidden)
						.activation("relu")
						.dropOut(0.1)
						.build())
				.layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
						.activation("identity")
						.nIn(nHidden).nOut(1).build())
				.pretrain(false).backprop(true).build()
				);

		criticBrain.init();

		// criticBrain = actorBrain.clone();
		// brain.setListeners(new ScoreIterationListener(1));

		expectedRewards = Nd4j.ones(actionCount);
		previousExpectedRewards = Nd4j.ones(actionCount);

		previousState = new float[weightCount];
		for (int w = 0; w < weightCount; w++) {
			previousState[w] = 0.0f;
		}
		previousStateNd = Nd4j.create(previousState);

	}

	public void act(int actionIndex) {
		if (isDebug)
		{
			// System.out.println("act(" + actionIndex + ")");
		}

		// Save action visit
		actionVisitCounts[actionIndex] += 1;
		// Switch from action index to motor action
		switch (actionIndex) {
			case 0:
				// leftAxis.enableMotor(true);
				// leftAxis.setMotorSpeed(-wristSpeed * 5);
				// rightAxis.enableMotor(true);
				// rightAxis.setMotorSpeed(-wristSpeed * 5);

				rightWristJoint.enableMotor(true);
				rightWristJoint.setMotorSpeed(wristSpeed);
				break;
			case 1:
				// leftAxis.enableMotor(true);
				// leftAxis.setMotorSpeed(wristSpeed * 5);
				// rightAxis.enableMotor(true);
				// rightAxis.setMotorSpeed(wristSpeed * 5);

				rightWristJoint.enableMotor(true);
				rightWristJoint.setMotorSpeed(-wristSpeed);
				break;
			case 2:
				rightArmJoint.enableMotor(true);
				rightArmJoint.setMotorSpeed(armSpeed);
				break;
			case 3:
				rightArmJoint.enableMotor(true);
				rightArmJoint.setMotorSpeed(-armSpeed);
				break;
			case 4:
				leftWristJoint.enableMotor(true);
				leftWristJoint.setMotorSpeed(wristSpeed);
				break;
			case 5:
				leftWristJoint.enableMotor(true);
				leftWristJoint.setMotorSpeed(-wristSpeed);
				break;
			case 6:
				leftArmJoint.enableMotor(true);
				leftArmJoint.setMotorSpeed(armSpeed);
				break;
			case 7:
				leftArmJoint.enableMotor(true);
				leftArmJoint.setMotorSpeed(-armSpeed);
				break;
			case 8:
				leftArmJoint.enableMotor(true);
				leftArmJoint.setMotorSpeed(0);
				break;
			case 9:
				// move leftArm down
				leftWristJoint.enableMotor(true);
				leftWristJoint.setMotorSpeed(0);
				break;
			case 10:
				rightArmJoint.enableMotor(true);
				rightArmJoint.setMotorSpeed(0);
				break;
			case 11:
				// move arm down
				rightWristJoint.enableMotor(true);
				rightWristJoint.setMotorSpeed(0);
				break;

		}

		// Save as previous action for next timestep
		previousAction = actionIndex;
		// previousActions.add(previousAction);

	}

	public void QUpdate(float delta) {
		// System.out.println("QFunctionUpdate");
		// Calculate the expected reward from the previous state and action, using critic brain
		float expectedReward = getExpectedReward(previousStateNd, previousAction, 0);

		// Get the current state
		float[] currentState = getState(previousAction);

		INDArray currentStateNd = Nd4j.create(currentState);

		// Get the immediate reward
		float immediateReward = getReward();

		// Get the expected rewards of the previous state
		// INDArray previousExpectedRewards = getExpectedRewards(previousStateNd, 0);
		// Get the expected rewards from the current state
		expectedRewards = getExpectedRewards(currentStateNd, 0);
		// float expectedReward = expectedRewards.getFloat(previousAction);

		// find max value action from the current state, used in the Q update for the previous state
		int maxActionIndex = 0;
		float maxActionValue = -10000;
		// For each action
		for (int a = 0; a < actionCount; a++) {

			// Calculate expected reward for action
			float actionReward = expectedRewards.getFloat(a);

			// Add the exploration bonus
			// exploreBonus = (float) Math.sqrt((2.0f * Math.log(updateCount)) / actionVisitCounts[a]);
			// actionReward += exploreBonus;
			// If expected action reward greater than max so far
			if (actionReward > maxActionValue) {
				// Save max action index and value
				maxActionIndex = a;
				maxActionValue = actionReward;
			}
		}

		// Get the value from the critic
		// float maxActionExpectedReward = getExpectedReward(currentStateNd, maxActionIndex, 0);
		float maxActionExpectedReward = maxActionValue;
		float reward = immediateReward + (futureDiscount * maxActionExpectedReward);

		if (previousStates.remainingCapacity() == 0) {
			// System.out.println("Learn");
			DataSetIterator iterator = getActorTrainingData(memoryCount, rng);

			actorBrain.fit(iterator);

			valueScore = (float) actorBrain.score();

			// Train critc
			// iterator = getCriticTrainingData(memoryCount, rng);

			// criticBrain.fit(iterator);
			updateCount += 1;
		}

		// Periodically copy actor to critic
		if (age.getStandardSeconds() % 30 == 0) {
			// criticBrain = actorBrain.clone();
		}

		// Get the next action as either the max action or a random action
		int nextAction = maxActionIndex;
		// If randomness is greater than a random roll
		if (randomness > Math.random()) {
			// Next action is random
			nextAction = (int) (actionCount * Math.random());
		}

		// System.out.println(nextAction);
		// If is not manual control
		if (!isManualControl) {
			// Take action
			act(nextAction);
		}

		// If is best reward so far
		if (reward > bestValue) {
			// Save best reward
			bestValue = reward;
			// Else if worst
		} else if (reward < worstValue) {
			// Save worst reward
			worstValue = reward;
		}

		// If close to best
		if (reward > bestValue * .25f) {
			// Reset time since good value
			timeSinceGoodValue = 0;
			// Else not close to best
		} else {
			// Increment time since good value
			timeSinceGoodValue += delta;
			// If over some threshold
			if (timeSinceGoodValue > 10) {
				reward -= timeSinceGoodValue * impatience;
			}
			// If over some threshold
			if (timeSinceGoodValue > 60) {
				resetMinMax();
				timeSinceGoodValue = 0;
				sendHome();
			}
		}

		if (timeSinceGoodValue > 10) {
			// Move randomness and learning rate toward max
			randomness = MyMath.lerp(randomness, maxRandomness, impatience);
			// learningRate = MyMath.lerp(learningRate, maxLearningRate, impatience);
		} else {
			// Move learning rate toward min
			// learningRate = MyMath.lerp(learningRate, minLearningRate, impatience);
			// Move randomness toward min
			randomness = MyMath.lerp(randomness, minRandomness, impatience);
		}

		// Create a target for the actor by setting the expected reward for the previous action
		// to the actual reward received in the current state
		float qUpdateValue = expectedReward
				+ (learningRate * (reward - expectedReward));
		// System.out.println("------------------------------------------------");
		// System.out.println("previousExpectedRewards Before: " + previousExpectedRewards.toString());
		previousExpectedRewards.putScalar(previousAction, qUpdateValue);
		// System.out.println("previousExpectedRewards After: " + previousExpectedRewards.toString());
		// System.out.println("------------------------------------------------");

		previousActions.add(previousAction);
		previousStates.add(previousStateNd);
		currentActorTarget.add(previousExpectedRewards);

		// System.out.println("------------------------------------------------");
		// System.out.println("Previous State: " + previousStates.toString());
		// System.out.println("currentActorTarget: " + currentActorTarget.toString());
		// System.out.println("------------------------------------------------");

		currentStates.add(currentStateNd);
		currentRewards.add(reward);

		// Save current value as previous for next step
		previousState = currentState;
		previousStateNd = currentStateNd;
		previousSpeed = speed;
		previousReward = reward;
		previousExpectedReward = expectedReward;
		previousExpectedRewards = expectedRewards;
		previousMaxActionValue = maxActionValue;
	}

	public float[] getState(int action) {
		// Get the current state [armAngle, wristAngle]
		// float armAngle = Math.round(
		// Math.abs(
		// Math.toDegrees(
		// armJoint.getJointAngle()
		// )
		// ) % 360
		// );
		float rightArmAngle = rightArmJoint.getJointAngle();
		// armAngle = MathUtils.clamp(armAngle, 0, armRange);

		// float wristAngle = (int) Math.round(
		// Math.abs(
		// Math.toDegrees(
		// wristJoint.getJointAngle())
		// ) % 360
		// );
		float rightWristAngle = rightWristJoint.getJointAngle();
		// float wristAngle = (int) Math.round(
		// Math.abs(
		// Math.toDegrees(
		// wristJoint.getJointAngle())
		// ) % 360
		// );
		// wristAngle = MathUtils.clamp(wristAngle, 0, wristRange);

		// Convert angle to [0.0-1.0] where 1.0 is the max angle
		rightArmAngle = MyMath.convertRanges((float) (rightArmAngle), 0.0f, (float) (Math.PI * 2f),
				0.0f,
				1.0f);
		rightWristAngle = MyMath.convertRanges((float) (rightWristAngle), 0.0f,
				(float) (Math.PI * 2f),
				0.0f,
				1.0f);

		float leftWristAngle = rightWristJoint.getJointAngle();
		float leftArmAngle = rightArmJoint.getJointAngle();
		// Convert angle to [0.0-1.0] where 1.0 is the max angle
		leftArmAngle = MyMath.convertRanges((float) (leftWristAngle), 0.0f, (float) (Math.PI * 2f),
				0.0f,
				1.0f);
		leftWristAngle = MyMath.convertRanges((float) (leftArmAngle), 0.0f, (float) (Math.PI * 2f),
				0.0f,
				1.0f);
		// Get the current state [armAngle, wristAngle]
		// float leftArmAngle = Math.round(
		// Math.abs(
		// Math.toDegrees(
		// leftArmJoint.getJointAngle()
		// )
		// ) % 360
		// );
		// armAngle = MathUtils.clamp(armAngle, 0, armRange);

		// float leftWristAngle = (int) Math.round(
		// Math.abs(
		// Math.toDegrees(
		// leftWristJoint.getJointAngle())
		// ) % 360
		// );
		// wristAngle = MathUtils.clamp(wristAngle, 0, wristRange);

		// Convert angle to [0.0-1.0] where 1.0 is the max angle
		// leftArmAngle = MyMath.convertRanges((float) (leftArmAngle), 0.0f, (float) (armRange),
		// -1.0f, 1.0f);
		// leftWristAngle = MyMath.convertRanges((float) (leftWristAngle), 0.0f, (float) (wristRange),
		// -1.0f,
		// 1.0f);

		float angleRatio = rightArmAngle / Math.max(rightWristAngle, 0.001f);

		float leftAngleRatio = leftArmAngle * leftWristAngle;
		// leftAngleRatio = MyMath.convertRanges((float) (leftAngleRatio), 0.0f, (float) (1/0),
		// -1.0f,
		// 1.0f);
		float bodyAngle = (float) body.getAngle();
		if (Math.abs(bodyAngle) > Math.PI * 2) {
			bodyAngle = (float) (bodyAngle % (Math.PI * 2));
		}
		bodyAngle = (float) (bodyAngle / Math.PI * 2);
		// bodyAngle = MyMath.convertRanges((float) (bodyAngle), 0.0f, 3.4f, -1.0f,
		// 1.0f);

		float xVelocity = body.getLinearVelocity().x;
		// Sigmoid the velocity
		// xVelocity = fastSig(xVelocity);
		if (xVelocity >= .5f) {
			xVelocity = 1;
		} else if (xVelocity < -.5f) {
			xVelocity = -1;
		} else {
			xVelocity = 0;
		}

		float y = body.getPosition().y / height;

		// Get the speed of each motor
		float armMotorSpeed = rightArmJoint.getJointSpeed();
		float wristMotorSpeed = rightWristJoint.getJointSpeed();
		armMotorSpeed = armMotorSpeed / armSpeed;
		wristMotorSpeed = wristMotorSpeed / armSpeed;
		// armMotorSpeed = MyMath.convertRanges((float) (armMotorSpeed), (float) (-armSpeed),
		// (float) (armSpeed),
		// -1.0f, 1.0f);
		// wristMotorSpeed = MyMath.convertRanges((float) (wristMotorSpeed), (float) (-wristSpeed),
		// (float) (wristSpeed), -1.0f, 1.0f);

		float leftArmMotorSpeed = leftArmJoint.getJointSpeed();
		float leftWristMotorSpeed = leftWristJoint.getJointSpeed();
		leftArmMotorSpeed = leftArmMotorSpeed / wristSpeed;
		leftWristMotorSpeed = leftWristMotorSpeed / wristSpeed;
		// leftArmMotorSpeed = MyMath.convertRanges((float) (leftArmMotorSpeed), -armSpeed,
		// (float) (armSpeed),
		// -1.0f, 1.0f);
		// leftWristMotorSpeed = MyMath.convertRanges((float) (leftWristMotorSpeed), -wristSpeed,
		// (float) (wristSpeed), -1.0f, 1.0f);

		// Save the current state
		float[] currentState = new float[] {
				rightArmAngle, rightWristAngle,
				armMotorSpeed, wristMotorSpeed,
				// angleRatio,
				// leftArmAngle, leftWristAngle,
				// leftArmMotorSpeed, leftWristMotorSpeed,
				// leftAngleRatio,
				// xVelocity,
				// bodyAngle,
				// y,
				// isTouchingGround,
				// 1.0f
		};

		return currentState;
	}

	public INDArray getExpectedRewards(INDArray state, int whichBrain) {
		INDArray rewards = null;
		if (whichBrain == 0) {
			rewards = actorBrain.output(state, false);
		} else if (whichBrain == 1) {
			rewards = actorBrain.output(state, false);
		}
		// final INDArray input = Nd4j.create(state, new int[] { 1, weightCount });
		// INDArray rewards = actorBrain.output(state, false);
		// brain.

		// for (int a = 0; a < actionCount; a++) {
		// float reward = 0;
		// // For each state feature and action
		// for (int x = 0; x < state.length; x++) {
		// // Add the feature value times the weight to the sum
		// reward += state[x] * weights[a][x];
		// }
		// }

		return rewards;
	}

	public float getReward() {
		// Get x velocity of body
		float xVelocity = body.getLinearVelocity().x;

		// If velocity is below threshold
		if (Math.abs(xVelocity) < 0.25f) {
			// Set to zero
			xVelocity = 0.0f;
			if (timeSinceGoodValue > 10) {
				xVelocity = -0.1f;
			}
		}
		// If adjusted velocity is 0
		// if (xVelocity == 0) {
		// xVelocity = (float) (-0.15f);
		// }

		// Calculate the speed as a moving average of the velocity
		speed = (1 - speedDecay) * speed + (speedDecay * xVelocity);
		acceleration = speed - previousSpeed;

		if (speed > maxSpeed) {
			maxSpeed = speed;
			bestSpeed = speed;
		}
		if (xVelocity < worstSpeed) {
			worstSpeed = speed;
		}

		// Get the value of the current state
		float reward = (speedValueWeight * speed) + (acceleration * averageSpeedValueWeight);
		// reward = MathUtils.clamp(reward, -1.0f, 1.0f);

		float bodyAngle = (float) body.getAngle();
		if (bodyAngle > Math.PI * 2) {
			bodyAngle = (float) (bodyAngle % (Math.PI * 2));
		}
		bodyAngle = (float) (-(Math.abs(bodyAngle) / (Math.PI * 2)) / 2.0);
		// bodyAngle = MyMath.convertRanges(bodyAngle, worstSpeed, bestSpeed, -1.0f, 1.0f);
		// float reward = MyMath.convertRanges(speed, worstSpeed, bestSpeed, -1.0f, 1.0f);

		// float reward =
		// Reduce by body angle
		// reward += bodyAngle;

		// Save current speed as previous for next step
		previousSpeed = speed;
		return reward;
	}

	public float getExpectedReward(INDArray state, int action, int whichBrain) {
		INDArray rewards = null;
		if (whichBrain == 0) {
			rewards = actorBrain.output(state, false);
		} else if (whichBrain == 1) {
			rewards = actorBrain.output(state, false);
		}
		// final INDArray input = Nd4j.create(state, new int[] { 1, weightCount });
		// INDArray rewards = actorBrain.output(state, false);
		// float reward = 0;
		// // For each state feature and action
		// for (int x = 0; x < state.length; x++) {
		// // Add the feature value times the weight to the sum
		// reward += state[x] * weights[action][x];
		// }

		return rewards.getFloat(action);
	}

	private ListDataSetIterator getActorTrainingData(int batchSize, Random rand) {

		INDArray inputs = Nd4j.vstack(previousStates);
		INDArray rewards = Nd4j.vstack(currentActorTarget);
		// System.out.println(inputs.toString());
		// System.out.println(rewards.toString());

		// Enrich memory
		// NdIndexIterator iter = new NdIndexIterator(memoryCount, actionCount);
		// double previousValue = 0;
		// while (iter.hasNext()) {
		// int[] nextIndex = iter.next();
		// // Get the action taken
		// int action = previousActions.
		// double nextVal = rewards.getDouble(nextIndex);
		// }

		DataSet dataSet = new DataSet(inputs, rewards);
		List<DataSet> listDs = dataSet.asList();
		Collections.shuffle(listDs, rng);
		return new ListDataSetIterator(listDs, batchSize);

	}

	private ListDataSetIterator getCriticTrainingData(int batchSize, Random rand) {
		INDArray inputsNd = Nd4j.vstack(currentStates);
		INDArray rewardsNd = Nd4j.vstack(currentStates);

		DataSet dataSet = new DataSet(inputsNd, rewardsNd);
		List<DataSet> listDs = dataSet.asList();
		Collections.shuffle(listDs, rng);
		return new ListDataSetIterator(listDs, batchSize);

	}

	public void update(float delta) {
		// System.out.println("update(" + delta + ")");
		// Update age
		age = age.plus((long) (delta * 1000));

		// Update timer
		updateTime += delta;

		// If time to update
		if (updateTime > updateTimer) {
			// Reset update timer
			updateTime = 0.0f;

			// If past the finish line
			if (body.getPosition().x > finishLine) {
				isPastFinish = true;

				resetMinMax();
				sendHome();
				// Call play's finishLine
				play.finishLine(this);
				return;
			}

			if (body.getPosition().y < -100) {
				sendHome();
			}

			// Perform Q Update
			QUpdate(delta);

			recoil();

		}

	}

	public void recoil() {
		float armAngle = previousState[0];
		float wristAngle = previousState[1];

		// // If outside range
		if (wristAngle <= -0.97f) {
			// if (rightWristJoint.getMotorSpeed() < 0) {
			// System.out.println("Error: wrist angle below 0");
			rightWristJoint.enableMotor(true);
			rightWristJoint.setMotorSpeed(wristSpeed * .05f);
			return;
			// }
		}
		if (wristAngle >= 0.97f) {
			// if (rightWristJoint.getMotorSpeed() > 0) {
			// System.out.println("Error: wrist angle above range");
			rightWristJoint.enableMotor(true);
			rightWristJoint.setMotorSpeed(-wristSpeed * .05f);
			return;
			// }

		}
		if (armAngle <= -0.97f) {
			// if (rightArmJoint.getMotorSpeed() < 0) {
			// System.out.println("Error: arm angle below 0");
			rightArmJoint.enableMotor(true);
			rightArmJoint.setMotorSpeed(armSpeed * .05f);
			return;
			// }
		}
		if (armAngle >= 0.97f) {
			// if (rightArmJoint.getMotorSpeed() > 0) {
			// System.out.println("Error: arm angle above range");
			rightArmJoint.enableMotor(true);
			rightArmJoint.setMotorSpeed(-armSpeed * .05f);
			return;
			// }
		}
		if (true) {
			return;
		}

		float leftArmAngle = leftArmJoint.getJointAngle();

		float leftWristAngle = leftWristJoint.getJointAngle();

		// If outside range
		if (leftWristAngle <= leftWristJoint.getLowerLimit() * 1.02f) {
			if (leftWristJoint.getMotorSpeed() < 0) {
				// System.out.println("Error: leftWrist angle below 0");
				leftWristJoint.enableMotor(true);
				leftWristJoint.setMotorSpeed(wristSpeed * .05f);
				return;
			}
		}
		if (leftWristAngle >= leftWristJoint.getUpperLimit() * 0.99f) {
			if (leftWristJoint.getMotorSpeed() > 0) {
				// System.out.println("Error: leftWrist angle above range");
				leftWristJoint.enableMotor(true);
				leftWristJoint.setMotorSpeed(-wristSpeed * .05f);
				return;
			}
		}
		if (leftArmAngle <= rightArmJoint.getLowerLimit() * 1.02f) {
			if (leftArmJoint.getMotorSpeed() < 0) {
				// System.out.println("Error: leftArm angle below 0");
				leftArmJoint.enableMotor(true);
				leftArmJoint.setMotorSpeed(armSpeed * .05f);
				return;
			}
		}
		if (leftArmAngle >= rightArmJoint.getUpperLimit() * .99f) {
			if (leftArmJoint.getMotorSpeed() > 0) {
				// System.out.println("Error: leftArm angle above range");
				leftArmJoint.enableMotor(true);
				leftArmJoint.setMotorSpeed(-armSpeed * .05f);
				return;
			}
		}
	}

	public void render(SpriteBatch spriteBatch, ShapeRenderer shapeRenderer,
			Camera camera,
			float delta, boolean showFinish) {

		// Get the color for the reward display
		Color color;

		float value = 0;
		if (previousReward > 0) {
			value = previousReward / bestValue;
		} else {
			value = previousReward / worstValue;
		}

		if (previousReward > 0) {
			// value = previousValue / bestValue;
			color = new Color(0f, value, 0f, 1);
		} else if (previousReward < 0) {
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

		// shapeRenderer.setProjectionMatrix(camera.combined);
		// shapeRenderer.begin(ShapeType.Line);
		// shapeRenderer.setColor(Color.BLUE);
		//
		// shapeRenderer.polygon(Vector2.bodyShape);
		// // shapeRenderer.rect(body.getPosition().x, body.getPosition().y - (height * .5f),1,height);
		// shapeRenderer.end();

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
				rightArmJoint.setLimits(0, (int) Math.toRadians(armRange));
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
				rightWristJoint.setLimits(0, (int) Math.toRadians(wristRange));
				GamePreferences.instance.wristRange = wristRange;
			}
		});
		tbl.add(sldWristRange).width(slideWidth);
		tbl.add(new Label("360", Assets.instance.skin));
		tbl.row();

		// Arm Torque Slider
		tbl.add(new Label("Arm Torque: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		final Slider sldArmTorque = new Slider(0, 10000, 1, false, Assets.instance.skin);
		sldArmTorque.setValue(armTorque);
		sldArmTorque.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				armTorque = (int) ((Slider) actor).getValue();
				rightArmJoint.setMaxMotorTorque(armTorque);
				GamePreferences.instance.armTorque = armTorque;
			}
		});
		tbl.add(sldArmTorque).width(slideWidth);
		tbl.add(new Label("10000", Assets.instance.skin));
		tbl.row();

		// Wrist Torque Slider
		tbl.add(new Label("Wrist Torque: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		final Slider sldWristTorque = new Slider(0, 10000, 1, false, Assets.instance.skin);
		sldWristTorque.setValue(wristTorque);
		sldWristTorque.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				wristTorque = (int) ((Slider) actor).getValue();

				rightWristJoint.setMaxMotorTorque(wristTorque);
				GamePreferences.instance.wristTorque = wristTorque;
			}
		});
		tbl.add(sldWristTorque).width(slideWidth);
		tbl.add(new Label("10000", Assets.instance.skin));
		tbl.row();

		// Arm Speed Slider
		tbl.add(new Label("Arm Speed: ", Assets.instance.skin));
		tbl.add(new Label("1", Assets.instance.skin));
		final Slider sldArmSpeed = new Slider(1, 10, 0.1f, false, Assets.instance.skin);
		sldArmSpeed.setValue(GamePreferences.instance.armSpeed);
		sldArmSpeed.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				armSpeed = (int) ((Slider) actor).getValue();
				rightArmJoint.setMotorSpeed(armSpeed);
				GamePreferences.instance.armSpeed = armSpeed;
			}
		});
		tbl.add(sldArmSpeed).width(slideWidth);
		tbl.add(new Label("10", Assets.instance.skin));
		tbl.row();

		// Wrist Speed Slider
		tbl.add(new Label("Wrist Speed: ", Assets.instance.skin));
		tbl.add(new Label("1", Assets.instance.skin));
		final Slider sldWristSpeed = new Slider(1, 10, 0.1f, false, Assets.instance.skin);
		sldWristSpeed.setValue(GamePreferences.instance.wristSpeed = wristSpeed);
		sldWristSpeed.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				wristSpeed = (int) ((Slider) actor).getValue();

				rightWristJoint.setMotorSpeed(wristSpeed);
				GamePreferences.instance.wristSpeed = wristSpeed;
			}
		});
		tbl.add(sldWristSpeed).width(slideWidth);
		tbl.add(new Label("10", Assets.instance.skin));
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

				// middleAxis.setSpringFrequencyHz(((Slider) actor).getValue());

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
		sldFriction.setValue(rightWrist.getFixtureList().first().getFriction());
		sldFriction.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = ((Slider) actor).getValue();
				GamePreferences.instance.friction = value;
				rightWrist.getFixtureList().first().setFriction(value);
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
	public void learnFrom(StandingCrate leader, float transferRate) {

		if (this.equals(leader) || this == leader) {
			return;
		}

		actorBrain = leader.actorBrain.clone();
		criticBrain = leader.criticBrain.clone();

		// Update weights
		// for (int a = 0; a < actionCount; a++) {
		// for (int x = 0; x < weights[0].length; x++) {
		// weights[a][x] = ((1 - transferRate) * weights[a][x])
		// + (transferRate * ((CrawlingCrate) leader).weights[a][x]);
		// }
		// }
	}

	public StandingCrate spawn(World world) {
		StandingCrate child = new StandingCrate(this.play);
		Vector2 position = body.getPosition();

		child.isDebug = false;
		float mutatedWidth = width
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f) * (mutationRate * 2.0 * width * Math
						.random()));
		float mutatedHeight = height
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f) * (mutationRate * 2.0 * height * Math
						.random()));
		float mutatedRideHeight = rideHeight
				+ (float) ((Math.random() > 0.5f ? 1.0f : -1.0f) * (mutationRate * 0.01f
						* rideHeight * Math
							.random()));

		float mutatedUpdateTimer = updateTimer
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f) * (mutationRate * updateTimer * Math
						.random()));
		float mutatedDensity = density
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f) * (mutationRate * density * Math
						.random()));
		float mutatedMinRandomness = minRandomness
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f) * (mutationRate * minRandomness * Math
						.random()));
		float mutatedMaxRandomness = maxRandomness
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f) * (mutationRate * maxRandomness * Math
						.random()));
		float mutatedMinLearningRate = minLearningRate
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f) * (mutationRate * minLearningRate * Math
						.random()));
		float mutatedMaxLearningRate = maxLearningRate
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f) * (mutationRate * maxLearningRate * Math
						.random()));
		float mutatedArmSpeed = armSpeed
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f) * (mutationRate * armSpeed * Math
						.random()));
		float mutatedWristSpeed = wristSpeed
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f) * (mutationRate * wristSpeed * Math
						.random()));
		float mutatedArmTorque = armTorque
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f) * (mutationRate * armTorque * Math
						.random()));
		float mutatedWristTorque = wristTorque
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f) * (mutationRate * wristTorque * Math
						.random()));
		float mutatedFutureDiscount = futureDiscount
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f)
				* (mutationRate * futureDiscount * Math.random()));
		float mutatedSpeedWeight = speedValueWeight
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 0.1 * speedValueWeight * Math.random()));
		float mutatedAverageSpeedWeight = averageSpeedValueWeight
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 0.1 * averageSpeedValueWeight * Math.random()));

		float mutatedLegSpread = legSpread
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 2.0 * legSpread * Math.random()));

		float mutatedWheelRadius = wheelRadius
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 1.5 * wheelRadius * Math.random()));

		float mutatedSuspension = suspension
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 0.5 * suspension * Math.random()));

		float mutatedArmLength = armLength
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 2.0 * armLength * Math.random()));

		float mutatedArmWidth = armWidth
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 2.0 * armWidth * Math.random()));

		float mutatedWristLength = wristLength
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 2.0 * armLength * Math.random()));

		float mutatedWristWidth = wristWidth
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 2.0 * wristWidth * Math.random()));

		float mutatedFootWidth = footWidth
				+ (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f)
				* (mutationRate * 2.0 * footWidth * Math.random()));

		// for (int index = 0; index < bodyShape.length; index++) {
		// bodyShape[index] = bodyShape[index]
		// + (float) ((Math.random() >= 0.5f ? 1.0f : -1.0f)
		// * (mutationRate * 0.01f * bodyShape[index] * Math.random()));
		// }

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
				mutatedArmTorque, mutatedWristTorque,
				bodyShape,
				mutatedDensity, mutatedLegSpread, mutatedWheelRadius, mutatedSuspension,
				mutatedRideHeight,
				mutatedArmLength, mutatedArmWidth, mutatedWristLength, mutatedWristWidth,
				mutatedFootWidth);

		return child;
	}

	public StandingCrate clone(World world) {
		StandingCrate child = new StandingCrate(this.play);
		Vector2 position = body.getPosition();

		// child.isDebug = true;

		child.init(world, position.x, position.y, width, height, maxRandomness, maxLearningRate,
				minRandomness, maxRandomness,
				minLearningRate, maxLearningRate, learningRateDecay, updateTimer, futureDiscount,
				speedValueWeight,
				averageSpeedValueWeight,
				armSpeed, wristSpeed, armRange, wristRange, armTorque, wristTorque, bodyShape,
				density);

		child.learnFrom(this, 1);

		return child;
	}

	@Override
	public boolean keyDown(int keycode) {
		// System.out.println("keyDown()");
		switch (keycode) {
			case Keys.UP:
				act(2);
				break;
			case Keys.DOWN:
				act(3);
				break;
			case Keys.RIGHT:
				act(0);
				break;
			case Keys.LEFT:
				act(1);
				break;

			case Keys.W:
				act(8);
				break;
			case Keys.A:
				act(6);
				break;
			case Keys.S:
				act(9);
				break;
			case Keys.D:
				act(7);
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
				if (rightArmJoint.getMotorSpeed() < 0) {
					rightArmJoint.enableMotor(holdMotors);
					rightArmJoint.setMotorSpeed(0);
				}
				break;
			case Keys.UP:
				if (rightArmJoint.getMotorSpeed() > 0) {
					rightArmJoint.enableMotor(holdMotors);
					rightArmJoint.setMotorSpeed(0);
				}
				break;
			case Keys.RIGHT:
				if (rightWristJoint.getMotorSpeed() > 0) {
					rightWristJoint.enableMotor(holdMotors);
					rightWristJoint.setMotorSpeed(0);
				}
				break;
			case Keys.LEFT:
				if (rightWristJoint.getMotorSpeed() < 0) {
					rightWristJoint.enableMotor(holdMotors);
					rightWristJoint.setMotorSpeed(0);
				}
				break;

			case Keys.W:
				if (leftArmJoint.getMotorSpeed() > 0) {
					leftArmJoint.enableMotor(holdMotors);
					leftArmJoint.setMotorSpeed(0);
				}
				break;
			case Keys.A:
				if (leftWristJoint.getMotorSpeed() < 0) {
					leftWristJoint.enableMotor(holdMotors);
					leftWristJoint.setMotorSpeed(0);
				}
				break;
			case Keys.S:
				if (leftArmJoint.getMotorSpeed() < 0) {
					leftArmJoint.enableMotor(holdMotors);
					leftArmJoint.setMotorSpeed(0);
				}
				break;
			case Keys.D:
				if (leftWristJoint.getMotorSpeed() > 0) {
					leftWristJoint.enableMotor(holdMotors);
					leftWristJoint.setMotorSpeed(0);
				}
				break;
		}
		return super.keyUp(keycode);
	}

	@Override
	public void sendHome() {
		bestValue = 0;
		worstValue = 0;

		System.out.println("sendHome()");
		for (Body body : bodies) {
			body.setTransform(new Vector2(0, height*.5f), (float) Math.toRadians(0));
			body.setLinearVelocity(new Vector2(0, 0));
		}

	};

	public void forget(float amount) {
		Layer[] layers = actorBrain.getLayers();
		// layers[0].

	}

	// Get a list of stats that will be displayed in the gui
	public ArrayList<String> getStats() {
		ArrayList<String> stats = new ArrayList<String>();
		// stats.add("Goal:" + goals[goal]);
		stats.add("Name:" + name);
		stats.add("Rank:" + rank);
		// stats.add("Finish:" + finishLine);
		stats.add("X:" + StringHelper.getDecimalFormat(body.getPosition().x, 0));
		if (age.getStandardHours() > 0) {
			stats.add("Age:" + age.getStandardHours() + "h");
		} else if (age.getStandardMinutes() > 0) {
			stats.add("Age:" + age.getStandardMinutes() + "m");
		} else {
			stats.add("Age:" + age.getStandardSeconds() + "s");
		}
		stats.add("Updates:" + updateCount);
		stats.add("NNScore:" + valueScore);
		// stats.add("Mutation Rate:" + mutationRate);

		// stats.add("Reward:" + String.format("%+1.6f", previousReward));
		// stats.add("Exploration Bonus:" + String.format("%+1.6f", previousExplorationValue));
		// stats.add("QValue:" + String.format("%+1.6f", previousValue));

		stats.add("Speed:" + String.format("%+1.2f", speed));
		stats.add("Acceleration:" + String.format("%+1.2f", acceleration));
		// stats.add("TouchingGround: " + isTouchingGround);
		// stats.add("Arm Angle:" + StringHelper.getDecimalFormat(previousArmAngle, 1));
		// stats.add("Wrist Angle:" + StringHelper.getDecimalFormat(previousWristAngle, 1));
		// stats.add("Angles:" + StringHelper.getDecimalFormat(previousArmAngle, 1) + ", "
		// + StringHelper.getDecimalFormat(previousWristAngle, 1));
		// stats.add("Body Angle:"
		// + StringHelper.getDecimalFormat((float) Math.toDegrees(body.getAngle()), 1));
		// stats.add("State = " + Arrays.toString(state));

		// stats.add("Best Value:" + String.format("%+1.6f", bestValue));
		// stats.add("Worst Value:" + String.format("%+1.6f", worstValue));
		stats.add("Max Speed:" + String.format("%+1.2f", maxSpeed));
		stats.add("TimeSinceGoodValue:" + String.format("%+1.6f", timeSinceGoodValue));

		stats.add("Action:" + previousAction
				+ ((previousAction != previousMaxAction) ? "*" : ""));

		// stats.add("Visited " + SACounts[previousArmAngle][previousWristAngle][previousAction]);
		if (previousState != null) {
			StringBuilder stateString = new StringBuilder();
			stateString.append("State : ");
			for (int x = 0; x < previousState.length; x++) {
				stateString.append(String.format("%+1.2f", previousState[x]));
				stateString.append("|");
			}
			stats.add(stateString.toString());

			stateString = new StringBuilder();
			if (currentActorTarget != null) {
				stateString.append("ExpectedRewards : ");
				for (int x = 0; x < expectedRewards.length(); x++) {
					stateString.append(String.format("%+1.2f", expectedRewards.getFloat(x)));
					stateString.append("|");
				}
			}
			stats.add(stateString.toString());

		}

		// stats.add("Explore Bonus:" + String.format("%+1.2f", exploreBonus));
		stats.add("Reward:" + String.format("%+1.2f", previousReward));
		stats.add("Expected Reward:" + String.format("%+1.2f", previousExpectedReward));
		stats.add("Value Range:" + String.format("%+1.2f", worstValue) + ", "
				+ String.format("%+1.2f", bestValue));

		stats.add("Actions:" + previousActions.toString());

		stats.add("Reward:" + String.format("%+1.2f", previousExpectedReward) + "->"
				+ String.format("%+1.2f", previousReward) + " ("
				+ String.format("%+1.2f", previousReward - previousExpectedReward) + ")");
		return stats;

	}

	public Body getBody() {
		return body;
	}

	// Save Q-State
	public void saveState() {
		saveState(locationToSave);
	}

	public void saveState(File locationToSave) {
		System.out.println("saveState( " + locationToSave.getAbsolutePath() + ")");
		try {
			// Save the model
			boolean saveUpdater = true;                                             // Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network
			// more in the future
			ModelSerializer.writeModel(actorBrain, locationToSave, saveUpdater);

			System.out.println("Done");

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void loadState() {
		loadState(locationToSave);
	}

	public void loadState(File locationToSave) {
		System.out.println("loadState " + locationToSave.getAbsolutePath());

		try {
			actorBrain = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
			criticBrain = actorBrain.clone();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public float getArmSpeed() {
		return armSpeed;
	}

	public void setArmSpeed(float armSpeed) {
		this.armSpeed = armSpeed;
		rightArmJoint.setMotorSpeed(armSpeed);
	}

	public float getWristSpeed() {
		return wristSpeed;
	}

	public void setWristSpeed(float wristSpeed) {
		this.wristSpeed = wristSpeed;
		rightWristJoint.setMotorSpeed(wristSpeed);
	}

	public int getArmRange() {
		return armRange;
	}

	public void setArmRange(int armRange) {
		this.armRange = armRange;
		rightArmJoint.setLimits(0, (float) Math.toRadians(armRange));
		GamePreferences.instance.armRange = armRange;
	}

	public int getWristRange() {
		return wristRange;
	}

	public void setWristRange(int wristRange) {
		this.wristRange = wristRange;
		rightWristJoint.setLimits(0, (float) Math.toRadians(wristRange));
		GamePreferences.instance.wristRange = wristRange;

	}

	public float getArmTorque() {
		return armTorque;
	}

	public void setArmTorque(float armTorque) {
		this.armTorque = armTorque;
		this.rightArmJoint.setMaxMotorTorque(armTorque);
	}

	public float getWristTorque() {
		return wristTorque;
	}

	public void setWristTorque(float wristTorque) {
		this.wristTorque = wristTorque;
		this.rightWristJoint.setMaxMotorTorque(wristTorque);
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

	public void resetMinMax() {
		bestValue = 1.0f;
		worstValue = -1.0f;

	}
}