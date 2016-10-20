package com.alec.walker;

import com.alec.Assets;
import com.alec.walker.Views.DirectedGame;
import com.alec.walker.Views.Play;
import com.alec.walker.Views.Transitions.ScreenTransition;
import com.alec.walker.Views.Transitions.ScreenTransitionFade;
import com.badlogic.gdx.assets.AssetManager;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.g2d.SpriteBatch;

public class Walker extends DirectedGame {
	
	@Override
	public void create () {
		Assets.instance.init(new AssetManager());
		ScreenTransition transition = 
				ScreenTransitionFade.init(0);

		setScreen(new Play(this), transition);
	}

}
