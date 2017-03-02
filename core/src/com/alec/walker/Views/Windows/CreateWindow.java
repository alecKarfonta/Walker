package com.alec.walker.Views.Windows;

import com.alec.Assets;
import com.alec.walker.Constants;
import com.alec.walker.Views.AbstractGameScreen;
import com.alec.walker.Views.Play;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.InputEvent;
import com.badlogic.gdx.scenes.scene2d.ui.*;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;
import com.badlogic.gdx.scenes.scene2d.utils.ClickListener;

public class CreateWindow extends Window {

	private AbstractGameScreen play;
	private Slider sldCount;
	
	public CreateWindow(AbstractGameScreen play, String title, Skin skin) {
		super(title, skin);
		this.play = play;
		this.init();
	}

	public void init() {

		this.setVisible(false);
		int padding = 10;

		Table tbl = new Table();
		tbl.row();

		tbl.add(new Label("Count: ", Assets.instance.skin));
		tbl.add(new Label("1", Assets.instance.skin));
		sldCount = new Slider(1, 10, 1, false, Assets.instance.skin);
		sldCount.setValue(5);
		sldCount.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
			}
		});
		tbl.add(sldCount);
		tbl.add(new Label("10", Assets.instance.skin));
		tbl.row();

		// Blocks
		TextButton btn = new TextButton("Block", Assets.instance.skin, "small");
		btn.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				int count = (int) sldCount.getValue();
				for (int index = 0; index < count; index++) {
					((Play) play).makeSomeBlocks(count);
				}

			}
		});
		tbl.add(btn).padRight(padding);
		tbl.add(btn).padLeft(padding);
//		tbl.row();
		
		// Balls
		btn = new TextButton("Balls", Assets.instance.skin, "small");
		btn.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				int count = (int) sldCount.getValue();
				for (int index = 0; index < count; index++) {
					((Play) play).makeSomeBalls(count);
				}

			}
		});
		tbl.add(btn).padRight(padding);
		tbl.add(btn).padLeft(padding);
		tbl.row();
		
		// Add One Legged Crate Button
		btn = new TextButton("One Leg Crate", Assets.instance.skin, "small");
		btn.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				// Pause rendering
				((Play) play).isRendering = false;
				int count = (int) sldCount.getValue();
				for (int index = 0; index < count; index++) {
					((Play) play).population.makeStandingCrate();
				}
				// Resume rendering
				((Play) play).isRendering = true;

			}
		});
		tbl.add(btn).padRight(padding);
		tbl.add(btn).padLeft(padding);
//		tbl.row();

		// Add Two Legged Crate Button
		btn = new TextButton("Two Leg Crate", Assets.instance.skin, "small");
		btn.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {

				// Pause rendering
				((Play) play).isRendering = false;
				int count = (int) sldCount.getValue();
				for (int index = 0; index < count; index++) {
					(((Play) play)).makeLeggedCrate();
				}
				// Resume rendering
				((Play) play).isRendering = true;
			}
		});
		tbl.add(btn).padRight(padding);
		tbl.add(btn).padLeft(padding);
//		tbl.row();
		
		this.add(tbl);

		this.pack();
		this.setPosition(
				Constants.VIEWPORT_GUI_WIDTH / 2 - (this.getWidth() / 2),
				Constants.VIEWPORT_GUI_HEIGHT);
		this.setMovable(true);
		this.setVisible(true);
		
		this.addListener(new ClickListener() {
			@Override
			public void clicked(InputEvent event, float x, float y) {
				event.cancel();
			}
		});
	}
	
	
}
