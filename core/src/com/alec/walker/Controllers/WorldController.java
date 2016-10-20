package com.alec.walker.Controllers;

import java.util.ArrayList;

import com.alec.walker.GamePreferences;
import com.alec.walker.Views.Play;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.physics.box2d.Body;
import com.badlogic.gdx.physics.box2d.BodyDef;
import com.badlogic.gdx.physics.box2d.Box2DDebugRenderer;
import com.badlogic.gdx.physics.box2d.ChainShape;
import com.badlogic.gdx.physics.box2d.FixtureDef;
import com.badlogic.gdx.physics.box2d.Shape;
import com.badlogic.gdx.physics.box2d.World;
import com.badlogic.gdx.physics.box2d.BodyDef.BodyType;
import com.badlogic.gdx.scenes.scene2d.Stage;
import com.badlogic.gdx.scenes.scene2d.ui.Table;
import com.badlogic.gdx.utils.Array;

public class WorldController {
	private static final String		TAG					= WorldController.class.getName();

	public World					world;
	private PopulationController	population;
	public float					width, height;
	private float					right;
	private float					left;
	private float					top;
	private float					bottom;
	public float					groundHeight;
	public int						zoom;

	private float					timestep;
	private int						velocityIterations;
	private int						positionIterations;

	public Body						groundBody;

	public ArrayList<Body>			destroyQueue;
	
	
	public WorldController() {
		// create the world with surface gravity
		world = new World(new Vector2(0f, -9.8f), true);
		world.setContactListener(new MyContactListener(this));

		destroyQueue = new ArrayList<Body>();
		zoom = 25;
		
		init();
	}

	public void init() {
		timestep = GamePreferences.instance.timestep;
		velocityIterations = GamePreferences.instance.positionIterations;
		positionIterations = GamePreferences.instance.velocityIterations;

		width = Gdx.graphics.getWidth();
		height = Gdx.graphics.getHeight();

		right = width / 2 / zoom;
		left = -(width / 2 / zoom);
		top = height / 2 / zoom;
		bottom = -(height / 2 / zoom);
		groundHeight = bottom + 10;

		destroyQueue = new ArrayList<Body>();

		createGround();
	}

	public void update(float delta) {
		// Update world
		world.step(timestep, velocityIterations, positionIterations);

		destroyQueue();
	}

	public void render(float delta) {

	}

	public void createGround() {

		BodyDef bodyDef = new BodyDef();
		FixtureDef fixtureDef = new FixtureDef();

		Shape shape = new ChainShape();

		// make a triangle
		for (int x = -1500; x < 1500; x++) {
			// body definition
			bodyDef.type = BodyType.StaticBody;
			bodyDef.position.set(x * 10, groundHeight - 3);
			bodyDef.allowSleep = false;
			// ground shape
			shape = new ChainShape();
			((ChainShape) shape).createChain(new Vector2[] {
					new Vector2(0 - 5, 0),
					new Vector2(0, 3),
					new Vector2(0 + 5, 0) });
			// fixture definition
			fixtureDef.shape = shape;
			fixtureDef.friction = .5f;
			fixtureDef.restitution = 0;
			// add the triangle to the world
			world.createBody(bodyDef).createFixture(fixtureDef);
		}

		bodyDef.type = BodyType.StaticBody;
		bodyDef.position.set(0, groundHeight);
		// clear the shape for the next chain
		shape = new ChainShape();

		// make the floor
		((ChainShape) shape).createChain(new Vector2[] { new Vector2(left - 10000, 0),
				new Vector2(right + 10000, 0) });

		// fixture definition
		fixtureDef.shape = shape;
		fixtureDef.friction = .5f;
		fixtureDef.restitution = 0;

		// add the floor to the world
		groundBody = world.createBody(bodyDef);
		groundBody.createFixture(fixtureDef);

		bodyDef.position.set(0, groundHeight - 3);
		// clear the shape for the next chain
		shape = new ChainShape();

		// make the floor
		((ChainShape) shape).createChain(new Vector2[] { new Vector2(left - 10000, 0),
				new Vector2(right + 10000, 0) });
		fixtureDef.shape = shape;
		world.createBody(bodyDef).createFixture(fixtureDef);

		bodyDef.position.set(0, groundHeight + 60);
		// clear the shape for the next chain
		shape = new ChainShape();

		// make the floor
		((ChainShape) shape).createChain(new Vector2[] { new Vector2(left - 100, 0),
				new Vector2(right + 100, 0) });

		world.createBody(bodyDef).createFixture(fixtureDef);

		bodyDef.position.set(0, groundHeight - 30);
		// clear the shape for the next chain
		shape = new ChainShape();

		// make the floor
		((ChainShape) shape).createChain(new Vector2[] { new Vector2(left - 250, 0),
				new Vector2(right + 250, 0) });

		fixtureDef.shape = shape;
		world.createBody(bodyDef).createFixture(fixtureDef);

	}
	


	// destroy stuff
	public void destroyBody(Body body) {
		// be sure the body you are trying to destroy is not already in the
		// queue
		if (!destroyQueue.contains(body))
			destroyQueue.add(body);
	}
	
	private void destroyQueue() {
		if (!destroyQueue.isEmpty()) {
			Gdx.app.debug(TAG, "destroy(): destroying queue");
			for (Body body : destroyQueue) {
				if (body != null) {
					world.destroyBody(body);
				}

			}
			destroyQueue.clear();
		}
	}
	
	
	public Vector2 getGravity() {
		return world.getGravity();
	}
	
	public void setGravity(Vector2 vector) {
		world.setGravity(vector);
	}
	

	public float getTimestep() {
		return timestep;
	}

	public void setTimestep(float timestep) {
		this.timestep = timestep;
	}

	public int getVelocityIterations() {
		return velocityIterations;
	}

	public void setVelocityIterations(int velocityIterations) {
		this.velocityIterations = velocityIterations;
	}

	public int getPositionIterations() {
		return positionIterations;
	}

	public void setPositionIterations(int positionIterations) {
		this.positionIterations = positionIterations;
	}
	
	

	
	
}
