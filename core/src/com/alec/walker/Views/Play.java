package com.alec.walker.Views;

import static com.badlogic.gdx.scenes.scene2d.actions.Actions.alpha;
import static com.badlogic.gdx.scenes.scene2d.actions.Actions.sequence;
import static com.badlogic.gdx.scenes.scene2d.actions.Actions.touchable;

import java.util.ArrayList;

import com.alec.Assets;
import com.alec.walker.Constants;
import com.alec.walker.GamePreferences;
import com.alec.walker.StringHelper;
import com.alec.walker.Controllers.BallFactory;
import com.alec.walker.Controllers.CameraController;
import com.alec.walker.Controllers.PopulationController;
import com.alec.walker.Controllers.WorldController;
import com.alec.walker.Models.BasicAgent;
import com.alec.walker.Models.BasicPlayer;
import com.alec.walker.Models.Car;
import com.alec.walker.Models.Crate;
import com.alec.walker.Models.CrawlingCrate;
import com.alec.walker.Models.LeggedCrate;
import com.alec.walker.Models.Player;
import com.alec.walker.Views.Windows.CreateWindow;
import com.alec.walker.Views.Windows.EvolutionWindow;
import com.alec.walker.Views.Windows.LearningWindow;
import com.alec.walker.Views.Windows.PhysicalWindow;
import com.alec.walker.Views.Windows.WorldOptionsWindow;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.Input.Keys;
import com.badlogic.gdx.InputAdapter;
import com.badlogic.gdx.InputMultiplexer;
import com.badlogic.gdx.InputProcessor;
import com.badlogic.gdx.graphics.Color;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.ShapeRenderer;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.math.Vector3;
import com.badlogic.gdx.physics.box2d.Body;
import com.badlogic.gdx.physics.box2d.Box2DDebugRenderer;
import com.badlogic.gdx.physics.box2d.Fixture;
import com.badlogic.gdx.physics.box2d.QueryCallback;
import com.badlogic.gdx.physics.box2d.joints.MouseJoint;
import com.badlogic.gdx.physics.box2d.joints.MouseJointDef;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.InputEvent;
import com.badlogic.gdx.scenes.scene2d.Stage;
import com.badlogic.gdx.scenes.scene2d.Touchable;
import com.badlogic.gdx.scenes.scene2d.ui.Skin;
import com.badlogic.gdx.scenes.scene2d.ui.Table;
import com.badlogic.gdx.scenes.scene2d.ui.TextButton;
import com.badlogic.gdx.scenes.scene2d.ui.Window;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;
import com.badlogic.gdx.scenes.scene2d.utils.ClickListener;

public class Play extends AbstractGameScreen {
	private static final String	TAG	= Play.class.getName();

	public Stage				stage;
	private Box2DDebugRenderer	debugRenderer;
	private InputAdapter		gameInput;
	private OrthographicCamera	camera;
	private OrthographicCamera	guiCamera;
	private CameraController	cameraHelper;
	private SpriteBatch			spriteBatch;
	private ShapeRenderer		shapeRenderer;
	private Table				mainTable;
	private Skin				skin;

	public PopulationController	population;
	public WorldController		world;
	public Player				player;
	private MouseJoint			mouseJoint;
	private Body				hitBody[];					// up to three bodies could be clicked at once
	private Body				tempBody;
	private Vector3				testPoint;
	private Vector2				dragPosition;

	public boolean				isPaused, isShowStats, isRendering, isRenderingAll;
	public boolean				isNaturalSelection;

	private BitmapFont			font;

	// GUI
	private CreateWindow		winCreate;
	private WorldOptionsWindow	winOptions;
	public EvolutionWindow		evolutionWindow;
	public LearningWindow		learningWindow;
	public PhysicalWindow		physicalWindow;
	private TextButton			btnWinOptSave;
	private InputMultiplexer	inputMultiplexer;
	private InputProcessor		menuInputProcessor;
	private GamePreferences		gamePreferences;

	public Play(DirectedGame game) {
		super(game);
		GamePreferences.instance.init();
		GamePreferences.instance.load();

		stage = new Stage();
		world = new WorldController();
		population = new PopulationController(this);

		isNaturalSelection = true;

		isRendering = true;
		isRenderingAll = false;
		isPaused = false;

		boolean drawBodies = true;
		boolean drawJoints = false;
		boolean drawAABBs = false;
		boolean drawInactiveBodies = true;
		boolean drawVelocities = false;
		boolean drawContacts = false;
		debugRenderer = new Box2DDebugRenderer(drawBodies, drawJoints, drawAABBs,
				drawInactiveBodies, drawVelocities, drawContacts);

		Gdx.gl.glClearColor(0, 0, 0, 1);

	}

	@Override
	public void render(float delta) {

		Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);

		// add each sprite

		// update the camera
		cameraHelper.setTarget(player.getBody().getPosition());
		cameraHelper.update(delta);
		cameraHelper.applyTo(camera);
		guiCamera.update();


		// Update Player
		if (player instanceof CrawlingCrate) {
			((CrawlingCrate) player).update(delta);
		} else {
			((LeggedCrate) player).update(delta);
		}
		// Render player
		// ((CrawlingCrate) player).render(spriteBatch, camera, delta);

		// Update each player
		int otherPlayerCount = population.allPlayers.size();
		// For each other player
		for (int index = 0; index < otherPlayerCount; index++) {
			BasicPlayer otherPlayer = population.allPlayers.get(index);
			// If is a crawling crate
			if (otherPlayer instanceof CrawlingCrate) {
				// If is rendering
				if ((isRendering && ((otherPlayer == player) || (isRenderingAll)))) {
					// Render
					boolean showFinish = (otherPlayer == player);
					((CrawlingCrate) otherPlayer).render(spriteBatch, shapeRenderer, camera, delta,
							showFinish);
				}
			} else if (otherPlayer instanceof LeggedCrate) {
				// If is rendering
				if ((isRendering && ((otherPlayer == player) || (isRenderingAll)))) {
					// Render
					((LeggedCrate) otherPlayer).render(spriteBatch, camera, delta);
				}
			}
		}


		if (isRendering) {
			// render the screen
			debugRenderer.render(world.world, camera.combined);
		}

		renderUI();

		update(delta);
		world.update(delta);

		// Update stage
		stage.act(delta);
		stage.draw();

	}

	public void renderUI() {

		spriteBatch.setProjectionMatrix(guiCamera.combined);
		spriteBatch.begin();

		// show the player's speed
		int count = 0;
		if (isShowStats) {

			font.draw(spriteBatch, "Population Age: " + population.getAge(),
					-(world.width * .5f) + 10,
					(world.height * .5f) - 5 - (20 * count));

			font.draw(spriteBatch, "Bot Count: " + population.allPlayers.size(),
					-(world.width * .5f) + 10,
					(world.height * .5f) - 25 - (20 * count));

			if (player != null && player.getStats() != null) {
				for (String value : player.getStats()) {
					font.draw(spriteBatch, value, -(world.width * .5f) + 10,
							(world.height * .5f) - 35 - (20 * (count + 1)));
					count += 1;
				}
			}
		}

		// fps
		if (isShowStats) {
			int fps = Gdx.graphics.getFramesPerSecond();
			if (fps >= 45) {
				font.setColor(Color.GREEN);
			} else if (fps >= 30) {
				font.setColor(Color.YELLOW);
			} else {
				font.setColor(Color.RED);
			}
			font.draw(spriteBatch, "FPS: " + fps, -(world.width * .5f) + 10,
					-(world.height * .5f) + 15);
			font.setColor(Color.WHITE);
		}
		spriteBatch.end();
	}

	public void update(float delta) {
		// Update timers

		// If player is agent
		if (player instanceof CrawlingCrate) {
			// Update the learning window with current player values
			learningWindow.update((CrawlingCrate) player);
		}
		

		// Render each player
		int otherPlayerCount = population.allPlayers.size();

		for (int index = 0; index < otherPlayerCount; index++) {
			try {
				BasicPlayer otherPlayer = population.allPlayers.get(index);
				// If is a crawling crate
				if (otherPlayer instanceof CrawlingCrate) {
					// If is not paused
					if (!isPaused) {
						// Update
						((CrawlingCrate) otherPlayer).update(delta);
					}
				} else if (otherPlayer instanceof LeggedCrate) {
					// If is not paused
					if (!isPaused) {
						// Update
						((LeggedCrate) otherPlayer).update(delta);
					}
				}
			} catch (Exception ex) {
				continue;
			}
		}

		population.update(delta);
	}

	@Override
	public void show() {
		// create a new stage object to hold all of the other objects
		camera = new OrthographicCamera(Constants.VIEWPORT_WIDTH,
				Constants.VIEWPORT_HEIGHT);

		shapeRenderer = new ShapeRenderer();
		shapeRenderer.setProjectionMatrix(camera.combined);
		spriteBatch = new SpriteBatch();
		spriteBatch.setProjectionMatrix(camera.combined);
		mainTable = new Table(skin);

		cameraHelper = new CameraController();
		spriteBatch = new SpriteBatch();

		// create a new table the size of the window
		mainTable = new Table(Assets.instance.skin);
		mainTable.setFillParent(true);
		isShowStats = GamePreferences.instance.isShowingStats;

		StringHelper.getInstance();
		StringHelper.init();

		hitBody = new Body[2];				// up to three bodies could be clicked at once
		testPoint = new Vector3();
		dragPosition = new Vector2();

		gameInput = new InputAdapter() {

			// Handle keyboard input
			@Override
			public boolean keyDown(int keycode) {
				switch (keycode) {
					case Keys.ESCAPE:
						break;
					case Keys.F:
						GamePreferences.instance.showFpsCounter = !GamePreferences.instance.showFpsCounter;
						break;
					case Keys.NUM_1:
						player = population.makeCrawlingCrate();
						break;
					case Keys.NUM_2:
						makeLeggedCrate();
						break;
					case Keys.NUM_3:
						makePlayerCar();
						break;
					case Keys.NUM_0:
						makeSomeBlocks(1);
						break;
					case Keys.PLUS:
					case Keys.PERIOD:
						changePlayer(population.nextPlayer());

						break;

					case Keys.MINUS:
					case Keys.COMMA:
						population.previousPlayer();
						break;

					case Keys.DEL:
					case Keys.END:
						population.removePlayer((BasicPlayer) player);

						break;

				}
				return false;
			}

			// zoom
			@Override
			public boolean scrolled(int amount) {
				cameraHelper.addZoom(cameraHelper.getZoom() * .25f * amount);
				return true;
			}

			QueryCallback	callback	= new QueryCallback() {
											@Override
											public boolean reportFixture(Fixture fixture) {
												// if the hit fixture's body is the ground body ignore it
												if (fixture.getBody() == world.groundBody)
													return true;

												if (fixture.testPoint(testPoint.x,
														testPoint.y)) {
													tempBody = fixture.getBody();
													return false;
												} else
													return true;
											}
										};

			// click or touch
			@Override
			public boolean touchDown(int screenX, int screenY, int pointer,
					int button) {
				// convert from vector2 to vector3
				testPoint.set(screenX, screenY, 0);
				// convert meters to pixel cords
				camera.unproject(testPoint);

				// reset the hit body
				hitBody[pointer] = null;

				// query the world for fixtures within a 2px box around the mouse click
				world.world.QueryAABB(callback, testPoint.x - 1.0f, testPoint.y - 1.0f,
						testPoint.x + 1.0f, testPoint.y + 1.0f);
				hitBody[pointer] = tempBody;

				// if something was hit
				if (hitBody[pointer] != null) {
					MouseJointDef mouseJointDef = new MouseJointDef();
					mouseJointDef.bodyA = world.groundBody; // ignored?
					mouseJointDef.bodyB = hitBody[pointer];
					mouseJointDef.collideConnected = true;
					mouseJointDef.target.set(hitBody[pointer].getPosition().x,
							hitBody[pointer].getPosition().y);
					mouseJointDef.maxForce = 3000.0f * hitBody[pointer].getMass();

					mouseJoint = (MouseJoint) world.world.createJoint(mouseJointDef);
					hitBody[pointer].setAwake(true);

					System.out.println("Body Clicked: " + hitBody[pointer].getUserData());
					if (hitBody[pointer].getUserData() != null) {
						try {
							try {
								player = (BasicAgent) hitBody[pointer].getUserData();
								population.selectPlayer(player);
								changePlayer(player);
							} catch (Exception ex) {

							}
						} catch (Exception ex) {
							ex.printStackTrace();
						}
					}
				}
				tempBody = null;
				return false;
			}

			@Override
			public boolean touchUp(int x, int y, int pointer, int button) {
				// if a mouse joint exists we simply destroy it
				if (mouseJoint != null) {
					world.world.destroyJoint(mouseJoint);
					mouseJoint = null;
				}
				return false;
			}

			@Override
			public boolean touchDragged(int x, int y, int pointer) {
				if (mouseJoint != null) {
					// convert from meters to pixels
					camera.unproject(testPoint.set(x, y, 0));
					// move the mouse joint to the new mouse location
					mouseJoint.setTarget(dragPosition.set(testPoint.x, testPoint.y));
				}
				return false;
			}

		};

		inputMultiplexer = new InputMultiplexer();

		inputMultiplexer.addProcessor(stage);

		inputMultiplexer.addProcessor(gameInput);


		createGUI();

		player = population.makeCrawlingCrate();
		
		changePlayer(player);
		

		if (player instanceof InputAdapter) {
			inputMultiplexer.addProcessor((InputProcessor) player);
		}
	}

	public void createGUI() {
		// Get the UI skin
		skin = new Skin(Gdx.files.internal("ui/uiskin.json"));

		// Get a font from assets
		font = Assets.instance.fonts.defaultSmall;
		font.getData().setScale(1);
		/* create some buttons */
		int padding = GamePreferences.instance.padding;
		;
		// Rank Button
		TextButton btnRank = new TextButton("Rank", Assets.instance.skin, "small");
		mainTable.add(btnRank).padRight(padding);
		btnRank.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				population.rank();
			}
		});
		

		// Pause Button
		TextButton btnWinOptStats = new TextButton("Stats", Assets.instance.skin, "small");
		mainTable.add(btnWinOptStats).padRight(padding);
		btnWinOptStats.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				isShowStats = !isShowStats;
				GamePreferences.instance.isShowingStats = isShowStats;
				GamePreferences.instance.save();
			}
		});
		TextButton btnWinOptPause = new TextButton("Pause", Assets.instance.skin, "small");
		mainTable.add(btnWinOptPause).padRight(padding);
		btnWinOptPause.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				isPaused = !isPaused;
			}
		});

		// Add Crawling Crate Button
		TextButton btn = new TextButton("Create", Assets.instance.skin, "small");
		btn.addListener(new ClickListener() {
			@Override
			public void clicked(InputEvent event, float x, float y) {
				showWindow(winCreate, !winCreate.isVisible());

				event.handle();
			}
		});
		mainTable.add(btn).padRight(padding);

		// Next Robot Button
		btn = new TextButton("Next", Assets.instance.skin, "small");
		btn.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				changePlayer(population.nextPlayer());
			}
		});
		mainTable.add(btn).padRight(padding);

		// Previous Robot Button
		btn = new TextButton("Previous", Assets.instance.skin, "small");
		btn.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				changePlayer(population.previousPlayer());
			}
		});
		mainTable.add(btn).padRight(padding);

		// Home All Button
		TextButton homeBtn = new TextButton("Home", Assets.instance.skin, "small");
		homeBtn.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				for (int index = 0; index < population.allPlayers.size(); index++) {
					population.allPlayers.get(index).sendHome();
				}
			}
		});
		mainTable.add(homeBtn).padRight(padding);

		// Reset Button
		TextButton btnReset = new TextButton("Reset Q", Assets.instance.skin, "small");
		mainTable.add(btnReset);
		btnReset.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {

				try {
					if (player instanceof CrawlingCrate) {
						((CrawlingCrate) player).initLearning();
					}
				}
				catch (Exception ex) {
					ex.printStackTrace();
				}
			}
		});

		// Manual Button
		TextButton manualBtn = new TextButton("Manual", Assets.instance.skin, "small");
		manualBtn.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				if (player instanceof BasicAgent) {
					((BasicAgent) player).setManualControl(((BasicAgent) player).getManualControl());
				}
			}
		});
		mainTable.add(manualBtn).padRight(padding);

		// Leader button
		TextButton btnLeader = new TextButton("Leader", Assets.instance.skin, "small");
		mainTable.add(btnLeader).padRight(padding);
		btnLeader.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				player = findLeader();
				population.selectPlayer(player);
				changePlayer(player);
			}
		});

		// Learning window toggle button
		TextButton btnLearn = new TextButton("Learning", Assets.instance.skin, "small");
		mainTable.add(btnLearn).padRight(padding);
		btnLearn.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				showWindow(learningWindow, !learningWindow.isVisible());
				event.handle();
			}
		});

		// Evolution window toggle button
		TextButton btnEvolution = new TextButton("Evolution", Assets.instance.skin, "small");
		mainTable.add(btnEvolution).padRight(padding);
		btnEvolution.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				showWindow(evolutionWindow, !evolutionWindow.isVisible());
				event.handle();
			}
		});

		// Physical window toggle button
		TextButton btnPhysical = new TextButton("Physical", Assets.instance.skin, "small");
		mainTable.add(btnPhysical).padRight(padding);
		btnPhysical.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				showWindow(physicalWindow, !physicalWindow.isVisible());
				event.handle();
			}
		});

		// Menu Button
		TextButton tbMenu = new TextButton("Menu", Assets.instance.skin, "small");
		// use an anonymous inner class for then click event listener
		tbMenu.addListener(new ClickListener() {
			@Override
			public void clicked(InputEvent event, float x, float y) {

				showWindow(winOptions, !winOptions.isVisible());

				event.handle();
			}
		});
		mainTable.add(tbMenu).padRight(padding);

		// Save Button
		btnWinOptSave = new TextButton("Save Settings", Assets.instance.skin, "small");
		mainTable.add(btnWinOptSave);
		btnWinOptSave.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				GamePreferences.instance.save();
			}
		});
		// Default
		btn = new TextButton("Default Settings", Assets.instance.skin, "small");
		btn.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				GamePreferences.instance.clear();
				GamePreferences.instance.init();
				GamePreferences.instance.load();

			}
		});
		mainTable.add(btn);

		// mainTable.padTop(25);
		// mainTable.padRight(25);
		mainTable.right();
		mainTable.bottom();

		winOptions = new WorldOptionsWindow(this, "Options", Assets.instance.skin);
		winOptions.addListener(new ClickListener() {
			@Override
			public void clicked(InputEvent event, float x, float y) {
				event.cancel();
			}
		});

		winOptions.bottom();
		winOptions.right();

		learningWindow = new LearningWindow(this, "Learning", Assets.instance.skin);
		if (player != null) {
			learningWindow.init((BasicAgent) player);
		}
		learningWindow.bottom();
		learningWindow.right();

		evolutionWindow = new EvolutionWindow(this, "Evolution", Assets.instance.skin);
		// if (player != null) {
		if (player instanceof CrawlingCrate) {
			evolutionWindow.init((CrawlingCrate) player);
		} else if (player instanceof LeggedCrate) {
			evolutionWindow.init((LeggedCrate) player);
		}

		// }
		// evolutionWindow.bottom();
		// evolutionWindow.right();
		evolutionWindow.center();
		evolutionWindow.setPosition(0, 200);

		physicalWindow = new PhysicalWindow("Physical", Assets.instance.skin);
		if (player != null) {
			physicalWindow.init(player);
		}
		physicalWindow.bottom();
		physicalWindow.right();

		winCreate = new CreateWindow(this, "Create", Assets.instance.skin);

		winCreate.top();
		// winCreate.right();

		// assemble stage for menu screen
		stage.clear();
		stage.addActor(mainTable);
		stage.addActor(winOptions);
		stage.addActor(winCreate);
		stage.addActor(evolutionWindow);
		stage.addActor(learningWindow);
		stage.addActor(physicalWindow);

		// Init GUI camera
		guiCamera = new OrthographicCamera(Constants.VIEWPORT_GUI_WIDTH,
				Constants.VIEWPORT_GUI_HEIGHT);

	}

	private void showWindow(Window window, boolean visible) {
		window.setVisible(visible);
		float alphaTo = 0.0f;
		float duration = 0.0f;
		Touchable touchEnabled;
		if (visible) {
			alphaTo = 1.0f;
			duration = 0.25f;
			touchEnabled = Touchable.enabled;
		} else {
			alphaTo = 0.0f;
			duration = 0.15f;
			touchEnabled = Touchable.disabled;
		}

		window.addAction(sequence(touchable(touchEnabled),
				alpha(alphaTo, duration)));
	}

	public void makeSomeBlocks(int count) {

		Vector2 playerPos = player.getBody().getPosition();

		// create some blocks
		for (int index = 0; index < count; index++) {
			Crate crate = new Crate(world.world,
					(float) (Math.random() * 3),
					(float) (Math.random() * 3),
					playerPos.x + 25f, (playerPos.y + 5f) + (20f * index),
					Constants.FILTER_CRATE, Constants.FILTER_CRATE | Constants.FILTER_CAR
							| Constants.FILTER_BOUNDARY);
			crate.getBody().applyLinearImpulse(new Vector2(25, 50), crate.getBody().getPosition(),
					false);
		}

	}

	public void makeSomeBalls(int count) {

		Vector2 playerPos = player.getBody().getPosition();
		// create some blocks
		for (int index = 0; index < count; index++) {
			float x = playerPos.x + 25f;
			float y = (playerPos.y + 5f) + (20f * index);
			BallFactory.instance.createBall(world.world, x, y, (float) (Math.random()),
					Constants.FILTER_BOUNDARY, (short) (Constants.FILTER_CRATE
							| Constants.FILTER_CAR
							| Constants.FILTER_BOUNDARY));
		}

	}

	public void makePlayerCar() {

		player = new Car(world.world,
				0, world.groundHeight + 4, // (x,y)
				15, 4);	// width, height

		// handle the input
		Gdx.input.setInputProcessor(new InputMultiplexer(
				// anonymous inner class for screen specific input
				gameInput, (Car) player));	// second input adapter for the input multiplexer

	}

	public void allLearnFrom(CrawlingCrate teacher) {
		System.out.println("learnFrom()");

		if (teacher == null) {
			return;
		}

		for (Player player : population.allPlayers) {
			if (player instanceof CrawlingCrate) {

				if (((CrawlingCrate) player).name == teacher.name) {
					continue;
				}
				try {
					((CrawlingCrate) player).learnFromLeader(teacher,
							GamePreferences.instance.transferRate);
				} catch (Exception ex) {
					ex.printStackTrace();
				}
			}

		}

	}

	public CrawlingCrate findLeader() {
		// If only one player
		if (population.allPlayers.size() == 1) {
			// Return player
			return population.allPlayers.get(0);
		}

		// Find the player with the greatest X value
		float leaderX = -100000;

		CrawlingCrate leader = null;

		for (CrawlingCrate otherPlayer : population.allPlayers) {
			if (otherPlayer.getBody().getPosition().x > leaderX) {
				leaderX = otherPlayer.getBody().getPosition().x;

				leader = otherPlayer;
			}
		}

		return leader;
	}

	public void changePlayer(Player player) {
		System.out.println("changePlayer()");
		if (player == null) {
			return;
		}
		this.player = player;

		// handle the input
		if (player instanceof InputAdapter) {
			Gdx.input.setInputProcessor(new InputMultiplexer(
					// anonymous inner class for screen specific input
					stage, gameInput, (InputAdapter) player));	// second input adapter for the input multiplexer

		}
		try {
			if (player instanceof CrawlingCrate) {
				learningWindow.init((CrawlingCrate) player);
				physicalWindow.init((CrawlingCrate) player);
				evolutionWindow.init((CrawlingCrate) player);
			} else if (player instanceof LeggedCrate) {
				learningWindow.init((LeggedCrate) player);
				physicalWindow.init((LeggedCrate) player);
				if (evolutionWindow == null) {
					System.out.println("evolutionWindow is null");
				}
				evolutionWindow.init((LeggedCrate) player);
			}
		} catch (Exception ex) {
			ex.printStackTrace();
		}

	}

	public LeggedCrate makeLeggedCrate() {

		LeggedCrate crate = new LeggedCrate(world.world,
				(250 * (population.allPlayers.size() + 1)), world.groundHeight + 4, // (x,y)
				8, 4);
//		population.allPlayers.add(crate);

		return crate;

	}

	public void finishLine(CrawlingCrate player) {
		System.out.println("finishLine(" + player.name + ")");
		this.player = player;

		// Rank the population
		population.rank();

		// If natural selection is on
		if (isNaturalSelection) {
			// Save the current population size
			int populationSize = population.allPlayers.size();
			System.out.println("populationSize = " + populationSize);

			// Send the top three home
			for (int index = 0; index < 3; index++) {
				population.allPlayers.get(index).sendHome();
			}

			ArrayList<BasicPlayer> removeQueue = new ArrayList<>();
			for (int index = 3; index < populationSize; index++) {
				removeQueue.add(population.allPlayers.get(index));
			}
			for (BasicPlayer removePlayer : removeQueue) {
				population.removePlayer(removePlayer);
			}
//			
//
			CrawlingCrate first = (CrawlingCrate) population.allPlayers.get(0);
			CrawlingCrate second = (CrawlingCrate) population.allPlayers.get(1);
			CrawlingCrate third = (CrawlingCrate) population.allPlayers.get(2);
			System.out.println("top 3 = " + first.name + ", " + second.name + ", " + third.name);
//			System.out.println("populationSize = " + populationSize);
//
			System.out.println("Make 30 childrend = " + populationSize);
			for (int index = 0; index <= 30; index++) {
				CrawlingCrate child = population.spawnCrawlingCrate(first);

				// Child learn from parent
				child.learnFromLeader(first, GamePreferences.instance.transferRate);
			}
			for (int index = 0; index <= 18; index++) {
				CrawlingCrate child = population.spawnCrawlingCrate(second);

				// Child learn from parent
				child.learnFromLeader(second, GamePreferences.instance.transferRate);
			}
			for (int index = 0; index <= 10; index++) {
				CrawlingCrate child = population.spawnCrawlingCrate(third);
				
				// Child learn from parent
				child.learnFromLeader(third, GamePreferences.instance.transferRate);
			}
			cameraHelper.setPosition(0,0);
			
			
			first.isPastFinish = false;
		}
		

	}

	@Override
	public void resize(int width, int height) {
		// reset the camera size to the width of the window scaled to the zoom level
		camera.viewportWidth = width / world.zoom;
		camera.viewportHeight = height / world.zoom;
		guiCamera.viewportHeight = height;
		guiCamera.viewportWidth = width;
		world.width = width;
		world.height = height;

		stage.getViewport().update(width, height, true);
		// invalidate the table hierarchy for it to reposition elements
		mainTable.invalidateHierarchy();
	}

	@Override
	public void dispose() {
		GamePreferences.instance.save();
		world.world.dispose();
		debugRenderer.dispose();
		// stage.dispose();
		spriteBatch.dispose();
	}

	@Override
	public void hide() {
		dispose();
		// isRendering = false;
	}

	@Override
	public void pause() {
	}

	@Override
	public void resume() {

		// isRendering = true;
	}

	@Override
	public InputProcessor getInputProcessor() {

		return inputMultiplexer;
	}







}
