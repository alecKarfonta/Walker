package com.alec.walker.Views.Windows;

import com.alec.walker.GamePreferences;
import com.alec.walker.Models.CrawlingCrate;
import com.alec.walker.Models.LeggedCrate;
import com.alec.walker.Models.Player;
import com.badlogic.gdx.scenes.scene2d.InputEvent;
import com.badlogic.gdx.scenes.scene2d.InputListener;
import com.badlogic.gdx.scenes.scene2d.ui.Skin;
import com.badlogic.gdx.scenes.scene2d.ui.Table;
import com.badlogic.gdx.scenes.scene2d.ui.Window;

public class PhysicalWindow extends Window {
	
	Table tbl;
	Table playerTbl;
	
	public PhysicalWindow(String title, Skin skin) {
		super(title, skin);
	}

	public void init(Player player) {
		// Clear old
		//this.removeActor(tbl);
		
		this.removeActor(playerTbl);

		int padding = GamePreferences.instance.padding;
		int slideWidth = GamePreferences.instance.slideWidth;
		

		tbl = new Table();

		tbl.columnDefaults(0).padRight(padding);
		tbl.columnDefaults(1).padRight(padding);
		tbl.columnDefaults(2).padRight(padding);


		

		
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
		if (player instanceof CrawlingCrate) {
			playerTbl = ((CrawlingCrate)player).getPhysicalMenu();
		} else if (player instanceof LeggedCrate) {
			playerTbl = ((LeggedCrate)player).getPhysicalMenu();
			
		}
		
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

