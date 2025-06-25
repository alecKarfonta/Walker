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

public class LearningWindow extends Window {

	private Table	tbl;
	private Play	play;
	private Slider	sldLearningRate, sldRandomness, sldMutationRate;

	public LearningWindow(Play play, String title, Skin skin) {
		super(title, skin);
		this.play = play;
	}

	public void update(final BasicAgent agent) {
		sldLearningRate.setValue(agent.getLearningRate());
		sldRandomness.setValue(agent.getRandomness());
	}

	public void init(final BasicAgent agent) {

		// Clear old
		this.removeActor(tbl);

		// this.setVisible(false);
		int padding = GamePreferences.instance.padding;

		int slideWidth = GamePreferences.instance.slideWidth;

		tbl = new Table();

		// Min Randomness Slide
		tbl.add(new Label("Min Randomness: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		Slider sldMinRandomness = new Slider(0, 0.2f, 0.00001f, false, Assets.instance.skin);
		sldMinRandomness.setValue(agent.getMinRandomness());
		sldMinRandomness.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = ((Slider) actor).getValue();
				agent.setMinRandomness(value);
				GamePreferences.instance.minRandomness = value;
			}
		});
		tbl.add(sldMinRandomness).width(slideWidth);
		tbl.add(new Label("0.2", Assets.instance.skin));
		tbl.row();

		// sldRandomness Slide
		tbl.add(new Label("Randomness: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		sldRandomness = new Slider(0, 0.2f, 0.00001f, false, Assets.instance.skin);
		sldRandomness.setValue(agent.getRandomness());
		sldRandomness.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = ((Slider) actor).getValue();
				agent.setRandomness(value);
				GamePreferences.instance.randomness = value;
			}
		});

		tbl.add(sldRandomness).width(slideWidth);
		tbl.add(new Label("0.2", Assets.instance.skin));
		tbl.row();

		// Max Randomness Slide
		tbl.add(new Label("Max Randomness: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		final Slider sldMaxRandomness = new Slider(0, 0.2f, 0.00001f, false, Assets.instance.skin);
		sldMaxRandomness.setValue(agent.getMaxRandomness());
		sldMaxRandomness.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = ((Slider) actor).getValue();
				agent.setMaxRandomness(value);
				GamePreferences.instance.maxRandomness = value;
			}
		});
		tbl.add(sldMaxRandomness).width(slideWidth);
		tbl.add(new Label("0.2", Assets.instance.skin));
		tbl.row();

		// Min LearningRate Slide
		tbl.add(new Label("Min Learning Rate: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));

		final Slider sldMinLearningRate = new Slider(0, 0.2f, 0.0001f, false, Assets.instance.skin);
		sldMinLearningRate.setValue(agent.getMinLearningRate());
		sldMinLearningRate.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = ((Slider) actor).getValue();
				agent.setMinLearningRate(value);
				GamePreferences.instance.minLearningRate = value;
			}
		});
		tbl.add(sldMinLearningRate).width(slideWidth);
		tbl.add(new Label("0.2", Assets.instance.skin));
		tbl.row();

		// LearningRate Slide
		tbl.add(new Label("Learning Rate: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));

		sldLearningRate = new Slider(0, 0.2f, 0.0001f, false, Assets.instance.skin);
		sldLearningRate.setValue(agent.getLearningRate());
		sldLearningRate.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = ((Slider) actor).getValue();
				agent.setLearningRate(value);
				GamePreferences.instance.learningRate = value;
			}
		});
		tbl.add(sldLearningRate).width(slideWidth);
		tbl.add(new Label("0.2", Assets.instance.skin));
		tbl.row();

		// Max LearningRate Slide
		tbl.add(new Label("Max Learning Rate: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		final Slider sldMaxLearningRate = new Slider(0, 0.2f, 0.0001f, false, Assets.instance.skin);
		sldMaxLearningRate.setValue(agent.getMaxLearningRate());
		sldMaxLearningRate.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = ((Slider) actor).getValue();
				agent.setMaxLearningRate(value);
				GamePreferences.instance.maxLearningRate = value;
			}
		});
		tbl.add(sldMaxLearningRate).width(slideWidth);
		tbl.add(new Label("0.2", Assets.instance.skin));
		tbl.row();

		// Future Discount Slider
		tbl.add(new Label("Future Discount: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		final Slider sldDiscount = new Slider(0, 1.0f, 0.01f, false, Assets.instance.skin);
		sldDiscount.setValue(agent.getFutureDiscount());
		sldDiscount.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = (((Slider) actor).getValue());
				agent.setFutureDiscount(value);
				GamePreferences.instance.futureDiscount = value;
			}
		});
		tbl.add(sldDiscount).width(slideWidth);
		tbl.add(new Label("1", Assets.instance.skin));
		tbl.row();

		// Exploration bonus Slider
		tbl.add(new Label("Exploration Bonus: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		final Slider sldExplorationBonus = new Slider(0, 1.0f, 0.01f, false, Assets.instance.skin);
		sldExplorationBonus.setValue(agent.getExplorationBonus());
		sldExplorationBonus.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = (((Slider) actor).getValue());
				agent.setExplorationBonus(value);
				GamePreferences.instance.explorationBonus = value;
			}
		});
		tbl.add(sldExplorationBonus).width(slideWidth);
		tbl.add(new Label("1", Assets.instance.skin));
		tbl.row();

		// Update Timer Slider
		tbl.add(new Label("Update Timer: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		Slider slider = new Slider(0, 1, .001f, false, Assets.instance.skin);
		slider.setValue(GamePreferences.instance.updateTimer);
		slider.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				agent.setUpdateTimer(((Slider) actor).getValue());
				GamePreferences.instance.updateTimer = ((Slider) actor).getValue();
			}
		});
		tbl.add(slider).width(slideWidth);
		tbl.add(new Label("1", Assets.instance.skin));
		tbl.row();

		// Impatience Slider
		tbl.add(new Label("Impatience: ", Assets.instance.skin));
		tbl.add(new Label("0", Assets.instance.skin));
		slider = new Slider(0, 0.1f, .000001f, false, Assets.instance.skin);
		slider.setValue(GamePreferences.instance.impatience);
		slider.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				float value = ((Slider) actor).getValue();
				agent.setImpatience(value);
				GamePreferences.instance.impatience = value;
			}
		});
		tbl.add(slider).width(slideWidth);
		tbl.add(new Label("0.1", Assets.instance.skin));
		tbl.row();

		

		tbl.add(agent.getLearningMenu()).colspan(3);

		tbl.row();
		// isDebug Checkbox
		final CheckBox chbxIsDebug = new CheckBox("Debug", Assets.instance.skin);
		chbxIsDebug.setChecked(agent.getIsDebug());
		chbxIsDebug.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				agent.setIsDebug(((CheckBox) actor).isChecked());
			}
		});
		tbl.add(chbxIsDebug);

		// isManualControl Checkbox
		final CheckBox chbxIsLearning = new CheckBox("Manual Control", Assets.instance.skin);
		chbxIsLearning.setChecked(agent.getManualControl());
		chbxIsLearning.addListener(new ChangeListener() {
			@Override
			public void changed(ChangeEvent event, Actor actor) {
				agent.setManualControl(chbxIsLearning.isChecked());
			}
		});
		tbl.add(chbxIsLearning);


		tbl.row();

		this.add(tbl);

		// Let TableLayout recalculate widget sizes and positions
		this.pack();
		// Move options window to top right corner
		this.setPosition(
				Constants.VIEWPORT_GUI_WIDTH,
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
