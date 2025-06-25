package com.alec.walker.Views.Transitions;

import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;

public interface ScreenTransition {
	public float getDuration ();
    public void render (SpriteBatch batch,Texture currScreen, Texture nextScreen, float alpha);
    
}
