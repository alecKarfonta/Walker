package com.alec.walker.Views.Transitions;

import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.graphics.GL20;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;
import com.badlogic.gdx.math.Interpolation;
import com.badlogic.gdx.utils.Array;

// slice the new screen in vertical bars then slide them in from the top and bottom
public class ScreenTransitionSlice implements ScreenTransition {
	public static final int UP = 1;
	public static final int DOWN = 2;
	public static final int UP_DOWN = 3;
	private static final ScreenTransitionSlice instance = new ScreenTransitionSlice();
	private float duration;
	private int direction;
	private Interpolation easing;
	private Array<Integer> sliceIndex = new Array<Integer>();

	public static ScreenTransitionSlice init(float duration, int direction,
			int numSlices, Interpolation easing) {
		instance.duration = duration;
		instance.direction = direction;
		instance.easing = easing;
		// create shuffled list of slice indices which determines
		// the order of slice animation
		instance.sliceIndex.clear();
		for (int i = 0; i < numSlices; i++)
			instance.sliceIndex.add(i);
		instance.sliceIndex.shuffle();
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
		float x = 0;
		float y = 0;
		int sliceWidth = (int) (w / sliceIndex.size); // divide slices into
														// equal width
		Gdx.gl.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		Gdx.gl.glClear(GL20.GL_COLOR_BUFFER_BIT);
		batch.begin();
		// draw the old screen
		batch.draw(currScreen, 0, 0, 0, 0, w, h, 1, 1, 0, 0, 0,
				currScreen.getWidth(), currScreen.getHeight(), false, true);
		if (easing != null)
			alpha = easing.apply(alpha);
		// for each slice
		for (int i = 0; i < sliceIndex.size; i++) {
			// new x = index + sliceWidth
			x = i * sliceWidth;
			// vertical displacement using randomized
			// list of slice indices
			float offsetY = h
					* (1 + sliceIndex.get(i) / (float) sliceIndex.size);
			switch (direction) {
			case UP:
				y = -offsetY + offsetY * alpha;
				break;
			case DOWN:
				y = offsetY - offsetY * alpha;
				break;
			case UP_DOWN:
				if (i % 2 == 0) {
					y = -offsetY + offsetY * alpha;
				} else {
					y = offsetY - offsetY * alpha;
				}
				break;
			}
			batch.draw(nextScreen, x, y, 0, 0, sliceWidth, h, 1, 1, 0, i
					* sliceWidth, 0, sliceWidth, nextScreen.getHeight(), false,
					true);
		}
		batch.end();
	}
}
