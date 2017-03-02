package com.alec.walker.desktop;

import com.alec.walker.Walker;
import com.badlogic.gdx.backends.lwjgl.LwjglApplication;
import com.badlogic.gdx.backends.lwjgl.LwjglApplicationConfiguration;

public class DesktopLauncher {
	public static void main (String[] arg) {
		LwjglApplicationConfiguration config = new LwjglApplicationConfiguration();
		config.width = 2400;
		config.height = 1200;
		new LwjglApplication(new Walker(), config);
	}
}
