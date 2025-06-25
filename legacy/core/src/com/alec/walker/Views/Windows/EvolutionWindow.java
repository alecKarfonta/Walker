package com.alec.walker.Views.Windows;

import java.util.ArrayList;

import com.alec.Assets;
import com.alec.walker.Constants;
import com.alec.walker.GamePreferences;
import com.alec.walker.Models.BasicAgent;
import com.alec.walker.Models.BasicPlayer;
import com.alec.walker.Models.CrawlingCrate;
import com.alec.walker.Views.Play;
import com.badlogic.gdx.scenes.scene2d.Actor;
import com.badlogic.gdx.scenes.scene2d.InputEvent;
import com.badlogic.gdx.scenes.scene2d.InputListener;
import com.badlogic.gdx.scenes.scene2d.ui.CheckBox;
import com.badlogic.gdx.scenes.scene2d.ui.Label;
import com.badlogic.gdx.scenes.scene2d.ui.Skin;
import com.badlogic.gdx.scenes.scene2d.ui.Slider;
import com.badlogic.gdx.scenes.scene2d.ui.Table;
import com.badlogic.gdx.scenes.scene2d.ui.TextButton;
import com.badlogic.gdx.scenes.scene2d.ui.Window;
import com.badlogic.gdx.scenes.scene2d.utils.ChangeListener;

public class EvolutionWindow extends Window {

	private Table	tbl;
	private Play	play;

	public EvolutionWindow(Play play, String title, Skin skin) {
		super(title, skin);
		this.play = play;
	}

	public void update(final BasicAgent agent) {
	}

	public void init(final BasicAgent agent) {

		// Clear old
		this.removeActor(tbl);

		// this.setVisible(false);
		int padding = GamePreferences.instance.padding;

		int slideWidth = GamePreferences.instance.slideWidth;

		tbl = new Table();
		tbl.row();
		// Mutation Rate Slide
		tbl.add(new Label("Mutation Rate: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		Slider sldMutationRate = new Slider(0, 0.25f, 0.0001f, false, Assets.instance.skin);
		sldMutationRate.setValue(GamePreferences.instance.mutationRate);
		sldMutationRate.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = ((Slider) actor).getValue();
				System.out.println("value = " + Float.toString(value));
				agent.setMutationRate(value);
				GamePreferences.instance.mutationRate = value;
			}
		});
		tbl.add(sldMutationRate).width(slideWidth);
		tbl.add(new Label(".5", Assets.instance.skin));
		tbl.row();
		
		// Finish Line Slide
		tbl.add(new Label("Finish Line: ", Assets.instance.skin));
		tbl.add(new Label("100", Assets.instance.skin));
		Slider sldFinishLine= new Slider(100, 10000, 100, false, Assets.instance.skin);
		sldFinishLine.setValue(((CrawlingCrate)(agent)).finishLine);
		sldFinishLine.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				int value = (int)((Slider) actor).getValue();
				System.out.println("value = " + Float.toString(value));
				for (BasicPlayer player : play.population.allPlayers) {
					((CrawlingCrate)(player)).finishLine = value;
				}
			}
		});
		tbl.add(sldFinishLine).width(slideWidth);
		tbl.add(new Label("10k", Assets.instance.skin));

		tbl.row();

		// Learn From Leader Button
		TextButton btnLearnLeader = new TextButton("Learn From Leader", Assets.instance.skin,
				"small");
		tbl.add(btnLearnLeader);
		btnLearnLeader.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				try {
					((Play) play).allLearnFrom(((Play) play).findLeader());
				}
				catch (Exception ex) {
					ex.printStackTrace();
				}
			}
		});

		// Learn From Selected Button
		TextButton btnLearnSelected = new TextButton("Learn From Selected", Assets.instance.skin,
				"small");
		tbl.add(btnLearnSelected);
		btnLearnSelected.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				try {
					((Play) play).allLearnFrom(agent);
				}
				catch (Exception ex) {
					ex.printStackTrace();
				}
			}
		});

		// Learn From Selected Button
		TextButton btnLearnAll = new TextButton("Learn From All", Assets.instance.skin, "small");
		tbl.add(btnLearnAll);
		btnLearnAll.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				try {
					((CrawlingCrate) agent).learnFromAll(((Play) play).population.allPlayers,
							agent.getLearningRate());
				}
				catch (Exception ex) {
					ex.printStackTrace();
				}
			}
		});

		tbl.row();


		// Spawn Button
		TextButton btnSpawn = new TextButton("Spawn", Assets.instance.skin, "small");
		tbl.add(btnSpawn);
		btnSpawn.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				try {
					play.population.spawnCrawlingCrate((CrawlingCrate)agent);
				}
				catch (Exception ex) {
					ex.printStackTrace();
				}
			}
		});

		// Clone Button
		TextButton btnClone = new TextButton("Clone", Assets.instance.skin, "small");
		tbl.add(btnClone);
		btnClone.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				try {
					play.population.cloneCrawlingCrate();
				}
				catch (Exception ex) {
					ex.printStackTrace();
				}
			}
		});

		// Delete Button
		TextButton btnDelete = new TextButton("Delete", Assets.instance.skin, "small");
		tbl.add(btnDelete);
		btnDelete.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				try {
					play.population.removePlayer(agent);
				}
				catch (Exception ex) {
					ex.printStackTrace();
				}
			}
		});

		// Delete Other Button
		TextButton btnDeleteOthers = new TextButton("Delete Others", Assets.instance.skin, "small");
		tbl.add(btnDeleteOthers);
		btnDeleteOthers.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				try {
					ArrayList<BasicPlayer> removeQueue = new ArrayList<>();
					for (BasicPlayer player : play.population.allPlayers) {
						if (player == agent) {
							continue;
						}
						removeQueue.add(player);
					}
					for (BasicPlayer player : removeQueue) {
						play.population.removePlayer(player);
					}
				}
				catch (Exception ex) {
					ex.printStackTrace();
				}
			}
		});
		tbl.row();

		this.add(tbl);

		// Let TableLayout recalculate widget sizes and positions
		this.pack();
		// Move options window to top right corner
		this.setPosition(
				0,
				200);
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
