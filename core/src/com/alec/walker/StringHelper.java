package com.alec.walker;

import com.alec.Assets;

import java.text.DecimalFormat;

public class StringHelper {
	public static final String TAG = Assets.class.getName();
	
	// singleton
	public static final StringHelper instance = new StringHelper();
	
	public final static  DecimalFormat[]		decimalFormats = new DecimalFormat[10];
	
	
	private StringHelper() {}
	
	
	public static void init() {
		String formatString = "#";
		for (int index = 0; index < 10; index++) {
			if (index == 1) {
				formatString += ".";
			}
			if (index > 0) {
				formatString += "#";
			}
			decimalFormats[index] = new DecimalFormat(formatString);
			
		}
	}
	
	public static StringHelper getInstance() {
		if (instance == null) {
			init();
		}
		return instance;
	}
	
	
	
	public static String getDecimalFormat(float value, int decimalPlaces) {
		return decimalFormats[decimalPlaces].format(value);
	}
	
}
