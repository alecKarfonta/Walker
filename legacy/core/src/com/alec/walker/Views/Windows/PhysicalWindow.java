package com.alec.walker.Views.Windows;

import com.alec.Assets;
import com.alec.walker.GamePreferences;
import com.alec.walker.Models.CrawlingCrate;
import com.badlogic.gdx.physics.box2d.Body;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.InputEvent;
import com.badlogic.gdx.scenes.scene2d.InputListener;
import com.badlogic.gdx.scenes.scene2d.ui.Label;
import com.badlogic.gdx.scenes.scene2d.ui.Skin;
import com.badlogic.gdx.scenes.scene2d.ui.Slider;
import com.badlogic.gdx.scenes.scene2d.ui.Table;
import com.badlogic.gdx.scenes.scene2d.ui.Window;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;

public class PhysicalWindow extends Window {
	
	Table tbl;
	Table playerTbl;
	
	public PhysicalWindow(String title, Skin skin) {
		super(title, skin);
	}

	public void init(CrawlingCrate crate) {
		final CrawlingCrate player = crate;
		// Clear old
		this.removeActor(tbl);
		
		this.removeActor(playerTbl);

		int padding = GamePreferences.instance.padding;
		int slideWidth = GamePreferences.instance.slideWidth;
		

		tbl = new Table();

		tbl.columnDefaults(0).padRight(padding);
		tbl.columnDefaults(1).padRight(padding);
		tbl.columnDefaults(2).padRight(padding);


		// Arm Speed Slider
		tbl.add(new Label("Arm Speed: ", Assets.instance.skin));
		tbl.add(new Label("0.01", Assets.instance.skin));
		final Slider sldArmSpeed = new Slider(0.01f, 4.0f, 0.001f, false, Assets.instance.skin);
		sldArmSpeed.setValue(player.getArmSpeed());
		sldArmSpeed.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				player.setArmSpeed(((Slider)actor).getValue());
				GamePreferences.instance.armSpeed = player.getArmSpeed();
			}
		});
		tbl.add(sldArmSpeed).width(slideWidth);
		tbl.add(new Label("4", Assets.instance.skin));
		tbl.row();

		// Wrist Speed Slider
		tbl.add(new Label("Wrist Speed: ", Assets.instance.skin));
		tbl.add(new Label("0.01", Assets.instance.skin));
		final Slider sldWristSpeed = new Slider(0.01f, 4.0f, 0.001f, false, Assets.instance.skin);
		sldWristSpeed.setValue(player.getWristSpeed());
		sldWristSpeed.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				player.setWristSpeed(((Slider)actor).getValue());
				GamePreferences.instance.wristSpeed = player.getWristSpeed();
			}
		});
		tbl.add(sldWristSpeed).width(slideWidth);
		tbl.add(new Label("4", Assets.instance.skin));
		tbl.row();

		// Arm Torque Slider
		tbl.add(new Label("Arm Torque: ", Assets.instance.skin));
		tbl.add(new Label("1", Assets.instance.skin));
		final Slider sldArmTorque = new Slider(1, 5000, 1, false, Assets.instance.skin);
		sldArmTorque.setValue(player.getArmTorque());
		sldArmTorque.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				player.setArmTorque(((Slider)actor).getValue());
				GamePreferences.instance.armTorque = player.getArmTorque();
			}
		});
		tbl.add(sldArmTorque).width(slideWidth);
		tbl.add(new Label("5000", Assets.instance.skin));
		tbl.row();

		// Arm Torque Slider
		tbl.add(new Label("Wrist Torque: ", Assets.instance.skin));
		tbl.add(new Label("1", Assets.instance.skin));
		final Slider sldWristTorque = new Slider(1, 5000, 1, false, Assets.instance.skin);
		sldWristTorque.setValue(player.getWristTorque());
		sldWristTorque.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				player.setWristTorque(((Slider)actor).getValue());
				GamePreferences.instance.wristTorque = player.getWristTorque();
			}
		});
		tbl.add(sldWristTorque).width(slideWidth);
		tbl.add(new Label("5000", Assets.instance.skin));
		tbl.row();

		
		// Gravity slide
//		tbl.add(new Label("Gravity: ", Assets.instance.skin));
//		tbl.add(new Label("-3", Assets.instance.skin));
//		final Slider sldGravity = new Slider(-3, 3, 1f, false,
//				Assets.instance.skin);
//		sldGravity.setValue(GamePreferences.instance.gravity);
//		sldGravity.addListener(new ChangeListener() {
//			@Override
//			public void changed(ChangeEvent event, Actor actor) {
//				GamePreferences.instance.gravity = ((Slider)actor).getValue();
//				for (Body body : player.getBodies()) {
//					body.setGravityScale(-((Slider)actor).getValue());
//				}
//			}
//		});
//		tbl.add(sldGravity).width(slideWidth);
//		tbl.add(new Label("3", Assets.instance.skin));
//		tbl.row();

		// Add default menu
		this.add(tbl);
		// Add player menu
		playerTbl = player.getPhysicalMenu();
		this.add(playerTbl).colspan(4);
//		this.bottom();
		this.right();
		// Let TableLayout recalculate widget sizes and positions
		this.pack();
		// Move options window to top right corner
		this.setPosition(
				0,
				25);
		this.setMovable(true);

		// Add listen for touch input to prevent interfacing with objects under the menu
		this.addListener(new InputListener() {
			public boolean touchDown(InputEvent event, float x, float y, int pointer, int button) {
				// Return true to say the input was handled
				return true;
			}
		});
	}
}

