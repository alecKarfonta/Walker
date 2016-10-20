package com.alec.walker.Views.Transitions;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.math.Interpolation;


// fade in the new screen over the old one
public class ScreenTransitionFade implements ScreenTransition {
	// singleton class
	private static final ScreenTransitionFade instance = new ScreenTransitionFade();
	private float duration;

	public static ScreenTransitionFade init(float duration) {
		instance.duration = duration;
		return instance;
	}

	@Override
	public float getDuration() {
		return duration;
	}

	@Override
	public void render(SpriteBatch batch, Texture currScreen,
			Texture nextScreen, float alpha) {
		float w = currScreen.getWidth();
		float h = currScreen.getHeight();
		// get the new alpha with a fade interpolation
		alpha = Interpolation.fade.apply(alpha);
		Gdx.gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
		batch.begin();
		
		// draw the old screen at full alpha
		batch.setColor(1, 1, 1, 1);
		batch.draw(currScreen, 
				0, 0, 
				0, 0, 
				w, h, 
				1, 1, 
				0, 0, 
				0,
				currScreen.getWidth(), currScreen.getHeight(), 
				false, true);
		
		// draw the new screen at the current alpha value
		batch.setColor(1, 1, 1, alpha);
		batch.draw(nextScreen, 
				0, 0, 
				0, 0, 
				w, h, 
				1, 1, 
				0, 0, 
				0,
				nextScreen.getWidth(), nextScreen.getHeight(), 
				false, true);
		batch.end();
	}

}
