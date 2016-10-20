package com.alec.walker.Views;

import com.alec.walker.Views.Transitions.ScreenTransition;
import com.badlogic.gdx.ApplicationListener;
import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.Pixmap.Format;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.graphics.glutils.FrameBuffer;

public abstract class DirectedGame implements ApplicationListener {
	private boolean init;
	private AbstractGameScreen currScreen;
	private AbstractGameScreen nextScreen;
	private FrameBuffer currFbo;
	private FrameBuffer nextFbo;
	private SpriteBatch batch;
	private ScreenTransition screenTransition;
	private float timer;

	public void setScreen(AbstractGameScreen screen) {
		setScreen(screen, null);
	}

	public void setScreen(AbstractGameScreen screen,
			ScreenTransition screenTransition) {
		int width = Gdx.graphics.getWidth();
		int height = Gdx.graphics.getHeight();
		// render to texture both screens
		if (!init) {
			currFbo = new FrameBuffer(Format.RGB888, width, height, false);
			nextFbo = new FrameBuffer(Format.RGB888, width, height, false);
			batch = new SpriteBatch();
			init = true;
		}
		// start new transition
		nextScreen = screen;
		nextScreen.show(); // activate next screen
		nextScreen.resize(width, height);
		nextScreen.render(0); // let screen update() once
		if (currScreen != null)
			currScreen.pause();
		nextScreen.pause();
		Gdx.input.setInputProcessor(null); // disable input
		this.screenTransition = screenTransition;
		timer = 0;
	}

	@Override
	public void render() {
		// semi-fixed time stepping:
		// constrain our deltaTime to 1/60 of a second to
		float deltaTime = Math.min(Gdx.graphics.getDeltaTime(), 1.0f / 60.0f);
		// if there is no nextScreen
		if (nextScreen == null) {
			// if there is a currScreen
			if (currScreen != null)
				// render it
				currScreen.render(deltaTime);
		} else {		// time left of transition
			float duration = 0;
			// if there is a screen transition, update it's time
			if (screenTransition != null)
				duration = screenTransition.getDuration();
			timer = Math.min(timer + deltaTime, duration);	// time left of transition
			// if the screen transition is over
			if (screenTransition == null || timer >= duration) {
				if (currScreen != null)				// hide the current screen
					currScreen.hide();
				nextScreen.resume();					// resume rendering of the new screen
				// enable input for next screen
				Gdx.input.setInputProcessor(nextScreen.getInputProcessor());
				currScreen = nextScreen;				// switch screens
				nextScreen = null;
				screenTransition = null;				// else the transition is not over
			} else {
				// render screens to FBOs
				currFbo.begin();
				if (currScreen != null)
					currScreen.render(deltaTime);
				currFbo.end();
				nextFbo.begin();
				nextScreen.render(deltaTime);
				nextFbo.end();
				// render transition effect to screen
				float alpha = timer / duration;
				screenTransition.render(batch, currFbo.getColorBufferTexture(),
						nextFbo.getColorBufferTexture(), alpha);
			}
		}
	}

	@Override
	public void resize(int width, int height) {
		if (currScreen != null)
			currScreen.resize(width, height);
		if (nextScreen != null)
			nextScreen.resize(width, height);
	}

	@Override
	public void pause() {
		if (currScreen != null)
			currScreen.pause();
	}

	@Override
	public void resume() {
		if (currScreen != null)
			currScreen.resume();
	}

	@Override
	public void dispose() {
		if (currScreen != null)
			currScreen.hide();
		if (nextScreen != null)
			nextScreen.hide();
		if (init) {
			currFbo.dispose();
			currScreen = null;
			nextFbo.dispose();
			nextScreen = null;
			batch.dispose();
			init = false;
		}
	}

}