package com.alec.walker.desktop;

import com.badlogic.gdx.tools.texturepacker.TexturePacker;
import com.badlogic.gdx.tools.texturepacker.TexturePacker.Settings;

public class TexturePack {

	private static boolean drawDebugOutline = false;
	
	public static void main(String[] args) {
		Settings settings = new Settings();
		settings.maxWidth = 4096;
		settings.maxHeight = 2048;
		settings.debug = drawDebugOutline;
		TexturePacker.process(settings, "assets-raw/", "../android/assets/ui", "ui.pack");
	}
}
