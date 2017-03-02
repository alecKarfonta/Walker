package com.alec.walker.Views.Windows;

import com.alec.Assets;
import com.alec.walker.Constants;
import com.alec.walker.GamePreferences;
import com.alec.walker.Models.BasicPlayer;
import com.alec.walker.Models.CrawlingCrate;
import com.alec.walker.Views.Play;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.InputEvent;
import com.badlogic.gdx.scenes.scene2d.InputListener;
import com.badlogic.gdx.scenes.scene2d.ui.*;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;

public class WorldOptionsWindow extends Window {

	private Play play;
	private Slider sldTimestep, sldGravityY, sldGravityX;
	private CheckBox chkIsRender;
	private Label gravityLabel, windLabel;
	
	
	public WorldOptionsWindow(Play play, String title, Skin skin) {
		super(title, skin);
		this.play = play;
		this.init();
	}

	public void init() {

		this.setVisible(false);
		
		int padding = 10;

		int slideWidth = 300;
		

		final Table tbl = new Table();

		// Timestep slide
		tbl.add(new Label("Timestep: ", Assets.instance.skin));
		tbl.add(new Label("1/240", Assets.instance.skin));
		sldTimestep = new Slider((1 / 240.0f), (1 / 5.0f), (1 / 600.0f), false,
				Assets.instance.skin);
		sldTimestep.setValue(GamePreferences.instance.timestep);
		sldTimestep.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				((Play)play).world.setTimestep(sldTimestep.getValue());
				GamePreferences.instance.timestep = ((Play)play).world.getTimestep();
			}
		});
		tbl.add(sldTimestep).width(slideWidth);
		tbl.add(new Label("1/5", Assets.instance.skin));
		tbl.row();
		

		// VelocityIterations slide
		tbl.add(new Label("Velocity Step: ", Assets.instance.skin));
		tbl.add(new Label("1", Assets.instance.skin));
		Slider sldVelocityIterations = new Slider(1, 50, 1, false,
				Assets.instance.skin);
		sldVelocityIterations.setValue(GamePreferences.instance.velocityIterations);
		sldVelocityIterations.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				GamePreferences.instance.velocityIterations = (int) ((Slider)actor).getValue();
				play.world.setVelocityIterations(GamePreferences.instance.velocityIterations);
			}
		});
		tbl.add(sldVelocityIterations).width(slideWidth);
		tbl.add(new Label("50", Assets.instance.skin));
		tbl.row();
		
		// PositionIterations slide
		tbl.add(new Label("Position Step: ", Assets.instance.skin));
		tbl.add(new Label("1", Assets.instance.skin));
		Slider sldPositionIterations = new Slider(1, 50, 1, false,
				Assets.instance.skin);
		sldPositionIterations.setValue(GamePreferences.instance.positionIterations);
		sldPositionIterations.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				GamePreferences.instance.positionIterations = (int) ((Slider)actor).getValue();
				play.world.setPositionIterations(GamePreferences.instance.positionIterations);
			}
		});
		tbl.add(sldPositionIterations).width(slideWidth);
		tbl.add(new Label("50", Assets.instance.skin));
		tbl.row();
		
		Vector2 gravity = play.world.getGravity();

		// GravityX slide
		windLabel = new Label("Wind: ", Assets.instance.skin);
		tbl.add(windLabel);
		tbl.add(new Label("-2", Assets.instance.skin));
		sldGravityX = new Slider(-2, 2, .001f, false,
				Assets.instance.skin);
		sldGravityX.setValue(gravity.x);
		sldGravityX.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				Vector2 gravity = play.world.getGravity();
				gravity.x = sldGravityX.getValue();
				windLabel.setText("Wind: " + gravity.x);
				((Play)play).world.setGravity(gravity);
			}
		});
		tbl.add(sldGravityX).width(slideWidth);
		tbl.add(new Label("2", Assets.instance.skin));
		tbl.row();
		

		// GravityY slide
		
		gravityLabel = new Label("Gravity: ", Assets.instance.skin);
		tbl.add(gravityLabel);
		tbl.add(new Label("-20", Assets.instance.skin));
		sldGravityY = new Slider(-20, 20, .5f, false,
				Assets.instance.skin);
		sldGravityY.setValue(gravity.y);
		sldGravityY.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				Vector2 gravity = play.world.getGravity();
				gravity.y = sldGravityY.getValue();
				gravityLabel.setText("Gravity: " + gravity.y);
				((Play)play).world.setGravity(gravity);
			}
		});
		tbl.add(sldGravityY).width(slideWidth);
		tbl.add(new Label("20", Assets.instance.skin));
		tbl.row();
		
		
		// IsRendering Checkbox
		tbl.add(new Label("Render", Assets.instance.skin));
		CheckBox chkIsRender = new CheckBox("World", Assets.instance.skin);
		chkIsRender.setChecked(true);
		chkIsRender.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				boolean value = ((CheckBox)actor).isChecked();
				play.isRendering = value;
			}
		});
		tbl.add(chkIsRender);
//		tbl.row();
		
		// IsRenderingAll Checkbox
		chkIsRender = new CheckBox("Rewards", Assets.instance.skin);
		chkIsRender.setChecked(true);
		chkIsRender.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				boolean value = ((CheckBox)actor).isChecked();
				play.isRenderingAll = value;
			}
		});
		tbl.add(chkIsRender);
		
		// IsRenderingNames Checkbox
		CheckBox chkIsRenderNames = new CheckBox("Names", Assets.instance.skin);
		chkIsRenderNames.setChecked(false);
		chkIsRenderNames.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				boolean value = ((CheckBox)actor).isChecked();
				
				for (BasicPlayer player : play.population.allPlayers) {
					if (player instanceof CrawlingCrate) {
						((CrawlingCrate) player).showName = value;
					}
				}
			}
		});
		tbl.add(chkIsRenderNames);
//		tbl.row();
		
		// Stats check box
		CheckBox btnWinOptStats = new CheckBox("Stats", Assets.instance.skin);
		tbl.add(btnWinOptStats).padRight(padding);
		btnWinOptStats.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				boolean value = ((CheckBox)actor).isChecked();
				((Play) play).isShowStats = value;
				GamePreferences.instance.isShowingStats = value;
			}
		});
		tbl.row();

		
		this.add(tbl);
		

		this.pack();
		// Move options window to top right corner
		this.setPosition(
				Constants.VIEWPORT_GUI_WIDTH,
				Constants.VIEWPORT_GUI_HEIGHT);
		this.setMovable(true);
		this.setVisible(true);

		// Add listen for touch input to prevent interfacing with objects under the menu
		this.addListener(new InputListener() {
			public boolean touchDown(InputEvent event, float x, float y, int pointer, int button) {
				// Return true to say the input was handled
				return true;
			}
		});
	}
}
