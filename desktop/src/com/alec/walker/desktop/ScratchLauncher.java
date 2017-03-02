package com.alec.walker.desktop;

import org.joda.time.Duration;

public class ScratchLauncher {
	public static void main (String[] arg) {
		
		
//		DateTime start = new DateTime(2004, 12, 25, 0, 0, 0, 0);
//		DateTime end = new DateTime(2005, 1, 1, 0, 0, 0, 0);
//
//		// duration in ms between two instants
		Duration dur = new Duration(2236 * 1000);

		// calc will be the same as end
//		DateTime calc = start.plus(dur);
		
//		System.out.println("calc = " + calc);
		System.out.println(dur.getStandardHours() + "h");
		System.out.println(dur.getStandardMinutes() + "m");
		System.out.println(dur.getStandardSeconds() + "s");
//		LwjglApplicationConfiguration config = new LwjglApplicationConfiguration();
//		config.width = 2400;
//		config.height = 1200;
//		LwjglApplication app = new LwjglApplication(new Walker(), config);
//		GamePreferences gamePreferences = GamePreferences.getInstance();
//		gamePreferences.init();
//		gamePreferences.load();
		
//		app.exit();
	}
}
