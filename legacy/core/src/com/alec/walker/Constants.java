package com.alec.walker;

import java.text.DecimalFormat;

public class Constants {
	public final static String	GameName			= "Walker";

	public final static float	TIMESTEP			= 1 / 60f;

	public static final float	VIEWPORT_WIDTH		= 240f;
	public static final float	VIEWPORT_HEIGHT		= 140f;
	public static final float	VIEWPORT_GUI_WIDTH	= 2400.0f;
	public static final float	VIEWPORT_GUI_HEIGHT	= 1400.0f;

	public static final String	PREFERENCES			= "default.prefs";
	
	
	public static final short			FILTER_BOUNDARY			= 0x0001;
	public static final short			FILTER_CAR					= 0x0002;
	public static final short			FILTER_CRATE				= 0x0003;
	public static final short			FILTER_RADAR				= 0x0020;
}


