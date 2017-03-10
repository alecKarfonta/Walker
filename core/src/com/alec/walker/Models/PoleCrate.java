package com.alec.walker.Models;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.joda.time.Duration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

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

//import org.nd4j.linalg.activations.Activation;

public class PoleCrate extends BasicAgent {
	public static final String		TAG					= PoleCrate.class.getName();

	// Keep a reference to play for convenience
	public Play						play;

	// Body parts
	public Body						body,
									leftWheel, rightWheel;
	public Body pole;

	public float					motorSpeed			= 10;
	// Joints: arm and wheels
	public WheelJoint				leftAxis, rightAxis;

	// Particle emitter for visualizing reward signal
	public ParticleEmitter			bodyTrail;
	public ParticleEmitter			finishParticleEmitter;
	public BitmapFont				font;
	// Size
	public float					width, height;
	public float[]					bodyShape;
	public float					density;

	public float					energyCapacity;
	public float					energy;
	public float					energyUsed;

	// Position
	// Arm parameters

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

	// Race
	public int						rank;
	public int						finishLine;

	public boolean					isPastFinish		= false;
	public boolean					isUsingEnergy		= false;

	public int						isTouchingGround	= 0;

	// State is each state property
	public float[]					state;
	// Save a weight for each feature
	public int						weightCount;
	public float[][]				weights;
	public float[]					previousState;
	public ArrayList<Experience>	experiences;
	public int						memoryCount;

	// QFunction Weights
	public float					speedValueWeight, averageSpeedValueWeight;

	// Control the reduction in state space
	public float					precision;

	public boolean					holdMotors			= true;
	public boolean					showName			= false;

	public MultiLayerNetwork		actorBrain;
	public MultiLayerNetwork		criticBrain;

	public INDArray					expectedRewards;
	public File						locationToSave		= new File("CrawlingCrate_Brain.zip");	// Where to save the network. Note: the file is in
																								// .zip format - can be opened
	// externally

	private INDArray				previousStateNd;
	// Random number generator seed, for reproducability
	public static final int			seed				= 12345;
	public static final int			iterations			= 1;
	public static final double		learningRateNN		= 0.00001f;
	// The range of the sample data, data in range (0-1 is sensitive for NN, you can try other ranges and see how it effects the results
	// also try changing the range along with changing the activation function
	public static final Random		rng					= new Random(seed);

	public PoleCrate(Play play) {
		this.play = play;

		rank = 0;
		finishLine = 1500;
		weightCount = 12;
		actionCount = 8;
		memoryCount = 10;
		experiences = new ArrayList<Experience>();

	}

	public void init(World world, float x, float y) {
		float defaultWidth = 10;
		float defaultHeight = 4;
		init(world, x, y, defaultWidth, defaultHeight);
	}

	public void init(World world, float x, float y,
			float width, float height) {

		speedValueWeight = GamePreferences.instance.speedValueWeight;
		averageSpeedValueWeight = GamePreferences.instance.speedValueWeight * 0.5f;

		learningRate = GamePreferences.instance.learningRate;

		futureDiscount = GamePreferences.instance.futureDiscount;

		float density = GamePreferences.instance.density;
		float updateTimer = GamePreferences.instance.updateTimer;

		init(world, x, y, width, height, randomness, learningRate, minRandomness, maxRandomness,
				minLearningRate, maxLearningRate, learningRateDecay, updateTimer, futureDiscount,
				speedValueWeight,
				averageSpeedValueWeight,
				bodyShape,
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
			float[] bodyShape, float density) {
		float defaultLegSpread = width * 12;
		float defaultWheelRadius = height / 2f;
		float defaultSuspension = GamePreferences.instance.suspension;
		float rideHeight = height * 0.5f;
		// Calculate arm lengths based on body size

		init(world, x, y, width, height, randomness, qlearningRate, minRandomness, maxRandomness,
				minlearningRate, maxLearningRate, learningRateDecay, updateTimer, futureDiscount,
				speedValueWeight, averageSpeedValueWeight, bodyShape, density, defaultLegSpread,
				defaultWheelRadius, defaultSuspension, rideHeight);
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
			float[] bodyShape, float density,
			float legSpread, float wheelRadius,
			float suspension, float rideHeight

			) {
		if (isDebug) {
			System.out.println("CrawlingCrate.init(world," + x + ", " + y + ", " + width + ", "
					+ height + ", " + randomness + ", " +
					qlearningRate + ", " + minlearningRate + ", " + maxLearningRate + ", "
					+ minlearningRate + ", " + maxLearningRate + ", "
					+ learningRateDecay + ", " + speedValueWeight + ", " + averageSpeedValueWeight
					+ ", bodyShape, " + density);
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
		// No limit
		// this.armRange = 360;
		// this.wristRange = 360;
		this.rideHeight = rideHeight;
		this.density = density;
		this.updateTimer = updateTimer;
		this.suspension = suspension;
		this.legSpread = legSpread;
		this.wheelRadius = wheelRadius;
		this.impatience = GamePreferences.instance.impatience;
		this.precision = 0.1f;

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
		fixtureDef.density = density * 1.0f;
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

		// create the wheels
		CircleShape wheelShape = new CircleShape();
		wheelShape.setRadius(wheelRadius * 0.5f);
		Sprite wheelSprite = new Sprite(new Texture("data/img/car/wheel.png"));
		wheelSprite.setSize(wheelShape.getRadius(), wheelShape.getRadius());
		wheelSprite.setOrigin(wheelSprite.getWidth() / 2, wheelSprite.getHeight() / 2);

		// Wheel fixture def
		FixtureDef wheelFixtureDef = new FixtureDef();
		wheelFixtureDef.density = density * 1.0f;
		wheelFixtureDef.friction = .1f;
		wheelFixtureDef.restitution = 0;
		wheelFixtureDef.filter.categoryBits = Constants.FILTER_CAR; // Interacts as
		wheelFixtureDef.filter.maskBits = Constants.FILTER_BOUNDARY; // Interacts with
		wheelFixtureDef.shape = wheelShape;

		leftWheel = world.createBody(bodyDef);
		leftWheel.createFixture(wheelFixtureDef);
		// leftWheel.setUserData(wheelSprite);

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

		// int numInput = weightCount;
		int numInput = weightCount;
		int numOutputs = actionCount;
		// int numOutputs = weightCount * actionCount + 1;
		int nHidden = 128;
		// Create the network
		actorBrain = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.learningRate(learningRateNN)
				.weightInit(WeightInit.XAVIER)
				.updater(Updater.NESTEROVS).momentum(0.9)
				.list()
				.layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
						.activation("tanh")
						// .dropOut(0.1)
						.build())

				.layer(1, new DenseLayer.Builder().nIn(nHidden).nOut(nHidden)
						.activation("tanh")
						// .dropOut(0.1)
						.build())
				.layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
						.activation("identity")
						.nIn(nHidden).nOut(numOutputs).build())
				.pretrain(false).backprop(true).build()
				);
		actorBrain.init();

		criticBrain = actorBrain.clone();
		// brain.setListeners(new ScoreIterationListener(1));

		expectedRewards = Nd4j.create(new float[] { 0, 0, 0, 0, 0, 0 });

		// weights = new float[actionCount][weightCount];

		// For each weight
		// for (int a = 0; a < actionCount; a++) {
		// for (int x = 0; x < weightCount; x++) {
		// // Init to a small random value
		// weights[a][x] = (float) (1.0f * Math.random());
		// }
		// }

		// for (int m = 0; m < memoryCount;)
		// experiences

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
		// Switch from action index to motor action
		switch (actionIndex) {
			case 0:
				leftAxis.enableMotor(true);
				leftAxis.setMotorSpeed(motorSpeed);
				rightAxis.enableMotor(true);
				rightAxis.setMotorSpeed(motorSpeed);
				break;
			case 1:
				leftAxis.enableMotor(true);
				leftAxis.setMotorSpeed(-motorSpeed);
				rightAxis.enableMotor(true);
				rightAxis.setMotorSpeed(-motorSpeed);
				break;
			case 2:
				leftAxis.enableMotor(true);
				leftAxis.setMotorSpeed(0);
				rightAxis.enableMotor(true);
				rightAxis.setMotorSpeed(0);
				break;

		}

		// Save as previous action for next timestep
		previousAction = actionIndex;
		previousActions.add(previousAction);

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

		// Get the expected rewards from the neural net
		expectedRewards = getExpectedRewards(previousStateNd, 0);
		// INDArray actorExpectedRewards = getExpectedRewards(previousStateNd, 0);
		// float expectedReward = expectedRewards.getFloat(previousAction);

		// find max value action from the current state, used in the Q update for the previous state
		int maxActionIndex = 0;
		float maxActionValue = -10000;
		// For each action
		for (int a = 0; a < actionCount; a++) {

			// Calculate expected reward for action
			// float actionReward = expectedRewards.getFloat(a);
			float actionReward = expectedRewards.getFloat(a);

			// If expected action reward greater than max so far
			if (actionReward > maxActionValue) {
				// Save max action index and value
				maxActionIndex = a;
				maxActionValue = actionReward;
			}
		}

		// Get the value from the critic
		float maxActionExpectedReward = getExpectedReward(previousStateNd, maxActionIndex, 1);
		float reward = immediateReward + (futureDiscount * maxActionExpectedReward);

		// Forget a random old memory
		if (experiences.size() > memoryCount) {
			int randomMemoryIndex = (int) Math.floor(Math.random() * experiences.size());
			experiences.remove(randomMemoryIndex);
		}

		// Create the target for the neural net where the action values are
		// the predicted values and the value for the previous action is the
		// current reward

		// Save experience
		Experience experience = new Experience(currentStateNd, previousStateNd, expectedRewards,
				previousAction,
				reward,
				expectedReward);
		experiences.add(experience);

		// Get a random experience
		// Experience randomExperience = experiences.get((int) Math.floor(Math.random()
		// * experiences.size()));

		// Calculate difference between the actual Q value (immediate reward plus max action reward)
		// and the Q function value
		// qDifference = randomExperience.reward - randomExperience.expectedReward;

		if (experiences.size() >= memoryCount) {
			// System.out.println("Learn");
			DataSetIterator iterator = getTrainingData(memoryCount, rng);

			actorBrain.fit(iterator);
			// criticBrain.fit(iterator);
		}

		// Periodically copy actor to critic
		if (age.getStandardSeconds() % 30 == 0) {
			criticBrain = actorBrain.clone();
		}

		// Update weights
		// for (int x = 0; x < randomExperience.previousState.length; x++) {
		// weights[randomExperience.action][x] = weights[randomExperience.action][x]
		// + (learningRate * qDifference * randomExperience.previousState[x]);
		// }
		// Update bias weight for all actions
		// for (int a = 0; a < actionCount; a++) {
		// weights[a][weights[0].length - 1] = weights[randomExperience.action][weights[0].length - 1];
		// }

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
			previousAction = nextAction;
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
			if (timeSinceGoodValue > 60) {
				resetMinMax();
				timeSinceGoodValue = 0;
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

		// Save current value as previous for next step
		previousState = currentState;
		previousStateNd = currentStateNd;
		previousSpeed = speed;
		previousReward = reward;
		previousExpectedReward = expectedReward;
		previousMaxActionValue = maxActionValue;
		if (isDebug) {
			previousActions.add(previousAction);
			previousValues.add(((int) reward));
		}
	}

	public float[] getState(int action) {

		float xVelocity = body.getLinearVelocity().x;

		// Save the current state
		float[] currentState = new float[] {
				xVelocity,
				1.0f
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

		return rewards;
	}

	public float getReward() {
		// Get x velocity of body
		float xVelocity = body.getLinearVelocity().x;

		// Calculate the speed as a moving average of the velocity
		speed = (1 - speedDecay) * speed + (speedDecay * xVelocity);
		acceleration = speed - previousSpeed;

		if (xVelocity > maxSpeed) {
			maxSpeed = xVelocity;
			bestSpeed = xVelocity;
		}
		if (xVelocity < worstSpeed) {
			worstSpeed = xVelocity;
		}

		float reward = MyMath.convertRanges(xVelocity, worstSpeed, bestSpeed, -1.0f, 1.0f);

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

		return rewards.getFloat(action);
	}

	private ListDataSetIterator getTrainingData(int batchSize, Random rand) {
		float[][] rewards = new float[memoryCount][actionCount];
		float[][] inputs = new float[memoryCount][weightCount];

		for (int i = 0; i < memoryCount; i++) {
			for (int a = 0; a < actionCount; a++) {
				// Set the value of each action to the expected reward
				rewards[i][a] = experiences.get(i).expectedRewards.getFloat(a);
			}
			// Set the reward of the previous action to the actual reward
			rewards[i][experiences.get(i).action] = experiences.get(i).reward;
			// Set the state features on the input
			for (int w = 0; w < weightCount; w++) {
				inputs[i][w] = experiences.get(i).previousState.getFloat(w);
			}

		}
		INDArray inputsNd = Nd4j.create(inputs);
		INDArray rewardsNd = Nd4j.create(rewards);

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
			if (body.getPosition().x > finishLine && isPastFinish == false) {
				isPastFinish = true;

				// Clear memory
				experiences = new ArrayList<>(memoryCount);

				resetMinMax();

				// Call play's finishLine
				// play.finishLine(this);
				// Reset energy
				energy = energyCapacity;
				return;
			}

			if (body.getPosition().y < -100) {
				sendHome();
			}

			// Perform Q Update
			QUpdate(delta);

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

		shapeRenderer.setProjectionMatrix(spriteBatch.getProjectionMatrix());
		// Show Energy bar
		shapeRenderer.begin(ShapeType.Filled);
		// shapeRenderer.setColor(Color.BLACK);
		// shapeRenderer.rect(body.getPosition().x, body.getPosition().y - (height * .5f),1,height);
		shapeRenderer.end();
		if (isUsingEnergy) {
			shapeRenderer.begin(ShapeType.Filled);
			shapeRenderer.setColor(Color.BLUE);
			shapeRenderer.rect(body.getPosition().x, body.getPosition().y - (height * .5f), 1,
					height
							* (float) ((float) energy / (float) energyCapacity));

			shapeRenderer.end();
		}

	}

	// getPhysicalMenu() - Get the part of the physical menu pertaining to the crawler
	public Table getPhysicalMenu() {
		int slideWidth = GamePreferences.instance.slideWidth;
		int padding = GamePreferences.instance.padding;

		Table tbl = new Table();
		tbl.columnDefaults(0).padRight(padding);
		tbl.columnDefaults(1).padRight(padding);
		tbl.columnDefaults(2).padRight(padding);

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
	public void learnFrom(PoleCrate leader, float transferRate) {

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


	public PoleCrate spawn(World world) {
		PoleCrate child = new PoleCrate(this.play);
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

		child.init(world,
				position.x, position.y,
				mutatedWidth, mutatedHeight,
				mutatedMaxRandomness, mutatedMaxLearningRate,
				mutatedMinRandomness, mutatedMaxRandomness,
				mutatedMinLearningRate, mutatedMaxLearningRate,
				learningRateDecay, mutatedUpdateTimer,
				mutatedFutureDiscount, mutatedSpeedWeight,
				mutatedAverageSpeedWeight,
				bodyShape,
				mutatedDensity, mutatedLegSpread, mutatedWheelRadius, mutatedSuspension,
				mutatedRideHeight);

		return child;
	}

	public PoleCrate clone(World world) {
		PoleCrate child = new PoleCrate(this.play);
		Vector2 position = body.getPosition();

		// child.isDebug = true;

		child.init(world, position.x, position.y, width, height, maxRandomness, maxLearningRate,
				minRandomness, maxRandomness,
				minLearningRate, maxLearningRate, learningRateDecay, updateTimer, futureDiscount,
				speedValueWeight,
				averageSpeedValueWeight,
				bodyShape,
				density);

		child.learnFrom(this, 1);

		return child;
	}

	@Override
	public boolean keyDown(int keycode) {
		// System.out.println("keyDown()");
		switch (keycode) {

			case Keys.RIGHT:
				act(0);
				break;
			case Keys.LEFT:
				act(1);
				break;
		}
		return super.keyDown(keycode);
	}

	@Override
	public boolean keyUp(int keycode) {
		// System.out.println("keyUp()");
		switch (keycode) {
			case Keys.RIGHT:
				act(2);
				break;
			case Keys.LEFT:
				act(2);
				break;
		}
		return super.keyUp(keycode);
	}

	@Override
	public void sendHome() {
		bestValue = 0;
		worstValue = 0;

		System.out.println("sendHome()");
		for (Body body : getBodies()) {
			body.setTransform(new Vector2(0, height * 2), (float) Math.toRadians(0));
			body.setLinearVelocity(new Vector2(0, 0));
		}

	};

	/* Gettets and Setters */
	@Override
	public ArrayList<Body> getBodies() {
		ArrayList<Body> bodies = new ArrayList<Body>();
		bodies.add(body);
		bodies.add(leftWheel);
		bodies.add(rightWheel);
		return bodies;
	}

	// Get a list of stats that will be displayed in the gui
	public ArrayList<String> getStats() {
		ArrayList<String> stats = new ArrayList<String>();
		// stats.add("Goal:" + goals[goal]);
		// stats.add("Name:" + name);
		// stats.add("Rank:" + rank);
		// stats.add("Finish:" + finishLine);
		stats.add("X:" + StringHelper.getDecimalFormat(body.getPosition().x, 0));
		if (age.getStandardHours() > 0) {
			stats.add("Age:" + age.getStandardHours() + "h");
		} else if (age.getStandardMinutes() > 0) {
			stats.add("Age:" + age.getStandardMinutes() + "m");
		} else {
			stats.add("Age:" + age.getStandardSeconds() + "s");
		}
		// stats.add("Mutation Rate:" + mutationRate);

		// stats.add("Reward:" + String.format("%+1.6f", previousReward));
		// stats.add("Exploration Bonus:" + String.format("%+1.6f", previousExplorationValue));
		// stats.add("QValue:" + String.format("%+1.6f", previousValue));

		if (isUsingEnergy) {
			stats.add("Energy:" + String.format("%+1.2f", energy) + " / "
					+ String.format("%+1.2f", energyCapacity));
			stats.add("Energy Used:" + String.format("%+1.2f", energyUsed));
		}
		stats.add("Speed:" + String.format("%+1.2f", speed));
		stats.add("Acceleration:" + String.format("%+1.2f", acceleration));
		stats.add("TouchingGround: " + isTouchingGround);
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
		// stats.add("Impatience:" + String.format("%+1.6f", impatience));
		// stats.add("Learning Rate:" + String.format("%+1.6f", learningRate));
		// stats.add("Randomness:" + String.format("%+1.6f", randomness));
		// stats.add("qDifference:" + String.format("%+1.6f", qDifference));

		// stats.add("MaxAction:" + previousMaxAction);
		// stats.add("MaxActionValue:" + String.format("%+1.6f", previousMaxActionValue));
		// stats.add("    Discounted:"
		// + String.format("%+1.6f", previousMaxActionValue * futureDiscount));

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
			if (expectedRewards != null) {
				stateString.append("ExpectedRewards : ");
				for (int x = 0; x < expectedRewards.columns(); x++) {
					stateString.append(String.format("%+1.2f", expectedRewards.getFloat(x)));
					stateString.append("|");
				}
			}
			stats.add(stateString.toString());

		}

		// StringBuilder weightString = new StringBuilder();
		// weightString.append("Weight :");
		// for (int x = 0; x < weights[previousAction].length; x++) {
		// weightString.append(String.format("%+1.2f", weights[previousAction][x]));
		// weightString.append("|");
		// }
		// stats.add(weightString.toString());
		stats.add("Reward:" + String.format("%+1.2f", previousReward));
		stats.add("Expected Reward:" + String.format("%+1.2f", previousExpectedReward));
		stats.add("Value Range:" + String.format("%+1.2f", worstValue) + ", "
				+ String.format("%+1.2f", bestValue));

		stats.add("Actions:" + previousActions.toString());
		// stats.add("QValues:" + previousValues.toString());
		// stats.add("Old QValue:" + String.format("%+1.6f", oldValue));
		// stats.add("New QValue:" + String.format("%+1.6f", newValue));
		// stats.add("QValue Delta:" + String.format("%+1.6f", newValue - oldValue));

		stats.add("Reward:" + String.format("%+1.2f", previousExpectedReward) + "->"
				+ String.format("%+1.2f", previousReward) + " ("
				+ String.format("%+1.2f", previousExpectedReward - previousReward) + ")");
		// stats.add("UpdateTimer:" + String.format("%+1.6f", updateTimer));

		// stats.add("Q-Values: ["
		// + String.format("%+1.2f", QValues[previousArmAngle][previousWristAngle][0])
		// + ", " + String.format("%+1.2f", QValues[previousArmAngle][previousWristAngle][1])
		// + ", "
		// + String.format("%+1.2f", QValues[previousArmAngle][previousWristAngle][2]) + ", "
		// + String.format("%+1.2f", QValues[previousArmAngle][previousWristAngle][3]) + ", "
		// + String.format("%+1.2f", QValues[previousArmAngle][previousWristAngle][4]) + ", "
		// + String.format("%+1.2f", QValues[previousArmAngle][previousWristAngle][5]) + " ] ");
		// stats.add("Action Visits: [" + SACounts[previousArmAngle][previousWristAngle][0]
		// + ", " + SACounts[previousArmAngle][previousWristAngle][1] + ", "
		// + SACounts[previousArmAngle][previousWristAngle][2] + ", "
		// + SACounts[previousArmAngle][previousWristAngle][3] + ", "
		// + SACounts[previousArmAngle][previousWristAngle][4] + ", "
		// + SACounts[previousArmAngle][previousWristAngle][5] + " ] ");
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
			// Gson gson = new Gson();
			// String content = gson.toJson(weights);
			//
			// File file = new File("CrawlingCrate_weights.txt");
			//
			// // if file doesnt exists, then create it
			// if (!file.exists()) {
			// file.createNewFile();
			// }
			//
			// FileWriter fw = new FileWriter(file.getAbsoluteFile());
			// BufferedWriter bw = new BufferedWriter(fw);
			// bw.write(content);
			// bw.close();

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

		// Gson gson = new Gson();
		// File file = new File("CrawlingCrate_weights.txt");
		// FileReader reader;
		// try {
		// reader = new FileReader(file);
		//
		// char[] chars = new char[(int) file.length()];
		// reader.read(chars);
		// String json = new String(chars);
		// reader.close();
		//
		// System.out.println(json);
		//
		// weights = gson.fromJson(json, float[][].class);
		// } catch (IOException e) {
		// e.printStackTrace();
		// }

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
		bestValue = 0;
		worstValue = 0;

	}
}