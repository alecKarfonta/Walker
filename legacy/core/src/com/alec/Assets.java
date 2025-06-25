package com.alec;


import com.badlogic.gdx.Gdx;
import com.badlogic.gdx.assets.AssetDescriptor;
import com.badlogic.gdx.assets.AssetErrorListener;
import com.badlogic.gdx.assets.AssetManager;
import com.badlogic.gdx.graphics.Texture;
import com.badlogic.gdx.graphics.Texture.TextureFilter;
import com.badlogic.gdx.graphics.g2d.BitmapFont;
import com.badlogic.gdx.graphics.g2d.TextureAtlas;
import com.badlogic.gdx.graphics.g2d.TextureAtlas.AtlasRegion;
import com.badlogic.gdx.scenes.scene2d.ui.Skin;
import com.badlogic.gdx.utils.Disposable;

public class Assets implements Disposable, AssetErrorListener {
	public static final String TAG = Assets.class.getName();
	public static final Assets instance = new Assets();	// singleton
	private AssetManager assetManager;
	private TextureAtlas atlas;
	
	private Assets() {}
	
	public Skin skin = new Skin(Gdx.files.internal("ui/uiskin.json"),
			new TextureAtlas("ui/uiskin.atlas"));

	public AssetFonts fonts;
	public AssetSounds sounds;
	public AssetMusic music;
	public AssetUI ui;
	
	public void init(AssetManager assetManager) {
		String textureAtlasPath = "ui/uiskin.atlas";
//		
		this.assetManager = assetManager;
		assetManager.setErrorListener(this);		
		assetManager.load(textureAtlasPath, TextureAtlas.class);
		assetManager.finishLoading();
		
		// load the sounds
		
		// log all the assets that were loaded
		Gdx.app.debug(TAG, "# of assets loaded: " + assetManager.getAssetNames().size);
		for (String asset : assetManager.getAssetNames()) {
			Gdx.app.debug(TAG, "asset: " + asset);
		}
		
		// load the texture atlas
		//atlas = assetManager.get(textureAtlasPath);
		atlas = skin.getAtlas();
		// enable texture filtering
		for (Texture texture : atlas.getTextures()) {
			texture.setFilter(TextureFilter.Linear, TextureFilter.Linear);
		}
		
		// create the game resources (inner Asset~ classes)
		fonts = new AssetFonts();
		sounds = new AssetSounds(assetManager);
		music = new AssetMusic(assetManager);
		ui = new AssetUI(atlas);
	}
	
	
	public class AssetUI {
		public AtlasRegion healthGaugeBorder, healthGaugeInfill, laserGaugeInfill;
		public AtlasRegion slider, sliderBar, sliderVert, sliderBarVert;
		public AssetUI (TextureAtlas atlas) {
			healthGaugeBorder = atlas.findRegion("healthGaugeBorder");
			healthGaugeInfill= atlas.findRegion("healthGaugeInfill");
			laserGaugeInfill = atlas.findRegion("laserGaugeInfill");			
		}
	}
	
	public class AssetSounds {

		public AssetSounds (AssetManager am) {
		}
	}
	
	public class AssetMusic {
		
		public AssetMusic (AssetManager am) {
		}
	}
	
	public class AssetFonts {
		public final BitmapFont defaultSmall;
		public final BitmapFont defaultNormal;
		
		public AssetFonts () {
			defaultSmall = new BitmapFont(Gdx.files.internal("fonts/white16.fnt"), false);
			defaultNormal = new BitmapFont(Gdx.files.internal("fonts/white32.fnt"), false);
			
//			defaultSmall.setScale(1f);
//			defaultNormal.setScale(.5f);
			
			defaultSmall.getRegion().getTexture().setFilter(TextureFilter.Linear, TextureFilter.Linear);
			defaultNormal.getRegion().getTexture().setFilter(TextureFilter.Linear, TextureFilter.Linear);

			defaultSmall.getData().setScale(0.5f);
		}
	}
	
	@Override
	public void error(AssetDescriptor asset, Throwable throwable) {
		Gdx.app.error(TAG, "Couldn't load asset: '" + asset.fileName + "' " + (Exception)throwable);
	}

	@Override
	public void dispose() {
		assetManager.dispose();
		fonts.defaultSmall.dispose();
		fonts.defaultNormal.dispose();
	}
	
	
}
