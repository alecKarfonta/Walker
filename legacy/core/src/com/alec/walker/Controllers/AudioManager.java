package com.alec.walker.Controllers;

import com.alec.walker.GamePreferences;
import com.badlogic.gdx.audio.Music;
import com.badlogic.gdx.audio.Sound;

public class AudioManager {
	public static final AudioManager instance = new AudioManager();
	private Music playingMusic;

	// singleton class
	private AudioManager() {
	}

	public void play(Sound sound) {
		play(sound, 1);
	}

	public void play(Sound sound, float volume) {
		play(sound, volume, 1);
	}

	public void play(Sound sound, float volume, float pitch) {
		play(sound, volume, pitch, 0);
	}

	public void play(Sound sound, float volume, float pitch, float pan) {
		if (!GamePreferences.instance.sound)
			return;
		sound.play(GamePreferences.instance.volSound * volume, pitch, pan);
	}
	
	public void loop(Sound sound, float volume, float pitch, float pan) {
		if (!GamePreferences.instance.sound)
			return;
		sound.loop(GamePreferences.instance.volSound * volume, pitch, pan);
	}
	
	public void stopSound(Sound sound) {
		sound.stop();
	}
	
	public void play(Music music) {
		stopMusic();
		playingMusic = music;
		if (GamePreferences.instance.music) {
			music.setLooping(true);
			music.setVolume(GamePreferences.instance.volMusic);
			music.play();
		}
	}

	public void stopMusic() {
		if (playingMusic != null)
			playingMusic.stop();
	}

	public void onSettingsUpdated() {
		if (playingMusic == null)
			return;
		playingMusic.setVolume(GamePreferences.instance.volMusic);
		if (GamePreferences.instance.music) {
			if (!playingMusic.isPlaying())
				playingMusic.play();
		} else {
			playingMusic.pause();
		}
	}
}