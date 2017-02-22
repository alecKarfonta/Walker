package com.alec.walker;

import java.lang.reflect.Field;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Preferences;
import com.badlogic.gdx.math.MathUtils;

public class GamePreferences {
	public static final String		TAG			= GamePreferences.class.getName();
	public static GamePreferences	instance	= new GamePreferences();
	public boolean					sound;
	public boolean					music;
	public float					volSound;
	public float					volMusic;
	public boolean					showFpsCounter;
	public boolean					useAccelerometer;

	public float					timestep;
	public int						positionIterations;
	public int						velocityIterations;
	public float					updateTimer;
	public float					gravity;

	public int						armRange;
	public int						wristRange;
	public float					armSpeed;
	public float					wristSpeed;
	public float					wristTorque;
	public float					armTorque;
	public float					suspension;
	public float					density;
	public float					friction;
	public float					linearDampening;

	public float					randomness;
	public float					minRandomness;
	public float					maxRandomness;
	public float					learningRate;
	public float					minLearningRate;
	public float					maxLearningRate;
	public float					futureDiscount;
	public float					explorationBonus;
	public float					impatience;
	public float					speedValueWeight;
	public float					mutationRate;
	public float					transferRate;
	
	
	// UI 
	public int						slideWidth;
	public int						padding;

	private Preferences				prefs;
	public boolean					useMonochromeShader;

	public boolean					isShowingStats;

	// singleton: prevent instantiation from other classes
	protected GamePreferences() {

	}

	public void init() {
		prefs = Gdx.app.getPreferences(Constants.PREFERENCES);
	}

	public void load() {
		System.out.println("GamePreferences.load()");

		// World Properties
		timestep = prefs.getFloat("timestep", (1 / 60f));
		gravity = prefs.getFloat("gravity", (-9.8f));
		updateTimer = prefs.getFloat("updateTimer", (0.1f));
		positionIterations = prefs.getInteger("positionIterations", 8);
		velocityIterations = prefs.getInteger("velocityIterations", 4);

		// Player Properties
		armRange = prefs.getInteger("armRange", (60));
		wristRange = prefs.getInteger("wristRange", (180));
		armSpeed = prefs.getFloat("armSpeed", (1));
		wristSpeed = prefs.getFloat("wristSpeed", (3));
		armTorque = prefs.getFloat("armTorque", (2000));
		wristTorque = prefs.getFloat("wristTorque", (4000));
		suspension = prefs.getFloat("suspension", (10));
		density = prefs.getFloat("density", (0.22f));
		friction = prefs.getFloat("friction", (0.33f));
		linearDampening = prefs.getFloat("linearDampening", (0.05f));

		// Learning Properties
		randomness = prefs.getFloat("randomness", (0.2f));
		minRandomness = prefs.getFloat("minRandomness", (0.001f));
		maxRandomness = prefs.getFloat("maxRandomness", (0.2f));
		learningRate = prefs.getFloat("learningRate", (0.01f));
		minLearningRate = prefs.getFloat("minLearningRate", (0.001f));
		maxLearningRate = prefs.getFloat("maxLearningRate", (1f));
		futureDiscount = prefs.getFloat("futureDiscount", (0.5f));
		explorationBonus = prefs.getFloat("explorationBonus", (100));
		impatience = prefs.getFloat("impatience", 0.0001f);
		speedValueWeight = prefs.getFloat("speedValueWeight", 1f);
		mutationRate = prefs.getFloat("mutationRate", 0.01f);
		transferRate = prefs.getFloat("transferRate", 0.10f);
		
		// Game Properties
		isShowingStats = prefs.getBoolean("isShowingStats", false);
		sound = prefs.getBoolean("sound", true);
		music = prefs.getBoolean("music", true);
		useAccelerometer = prefs.getBoolean("useAccelerometer", true);
		volSound = MathUtils
				.clamp(prefs.getFloat("volSound", 0.5f), 0.0f, 1.0f);
		volMusic = MathUtils
				.clamp(prefs.getFloat("volMusic", 0.5f), 0.0f, 1.0f);
		showFpsCounter = prefs.getBoolean("showFpsCounter", false);
		useMonochromeShader = prefs.getBoolean("useMonochromeShader", false);
		

		// UI Properties
		padding = prefs.getInteger("padding", 10);
		slideWidth = prefs.getInteger("slideWidth", 400);
	}

	// Save using reflection: each property is saved using the appropriate
	// write method based on it's name and type. 
	public void save() {
		System.out.println("GamePrefences.save()");

		Field[] fields = GamePreferences.class.getDeclaredFields();

		for (Field field : fields) {

			// Reflectively get the name and type of the field
			String propertyName = field.getName();
			String propertyType = field.getType().getName();

			// Skip some fields
			if (propertyName == "TAG" || propertyName == "instance" || propertyName == "prefs") {
				continue;
			}

			field.setAccessible(true);

			System.out.print("Saving : " + propertyName + " of type " + propertyType
					+ " with value = ");

			// Switch on the field type
			switch (propertyType) {
			// Float
				case "float":
					try {
						Float value = (Float) field.get(GamePreferences.instance);
						System.out.print(value);
						prefs.putFloat(propertyName, value);
					} catch (IllegalArgumentException | IllegalAccessException e) {
						e.printStackTrace();
					}
					break;
				// Boolean
				case "java.lang.Boolean":
				case "boolean":
					try {
						Boolean value = (Boolean) field.get(GamePreferences.instance);
						System.out.print(value);
						 prefs.putBoolean(propertyName, value);
					} catch (IllegalArgumentException | IllegalAccessException e) {
						e.printStackTrace();
					}
					break;
				// Integer
				case "int":
				case "java.lang.Integer":
				case "Integer":
					try {
						Integer value = (Integer)field.getInt(GamePreferences.instance);
						System.out.print(value);
						prefs.putInteger(propertyName, value);
					} catch (IllegalArgumentException | IllegalAccessException e) {
						e.printStackTrace();
					}
					break;
			}

			System.out.println();
		}

		System.out.println("------------------------------------------------------------------");

		prefs.flush();
	}

	public void clear() {
		prefs.clear();
	}
}