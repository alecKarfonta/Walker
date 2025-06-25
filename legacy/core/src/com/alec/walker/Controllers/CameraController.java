package com.alec.walker.Controllers;


import com.badlogic.gdx.graphics.OrthographicCamera;
import com.badlogic.gdx.math.MathUtils;
import com.badlogic.gdx.math.Vector2;
import com.badlogic.gdx.physics.box2d.Body;

public class CameraController {
	private static final String TAG = CameraController.class.getName();
	
	// settings
	public final float MAX_ZOOM_IN = .1f,
			MAX_ZOOM_OUT = 100f,
			FOLLOW_SPEED = .05f,
			ZOOM_SPEED = .05f;

	private Body followBody;
	private Vector2 position;	
	private Vector2 target;
	private Vector2 zoom;	// vectors just so you can lerp with it's function
	private Vector2 targetZoom;

	public CameraController() {
		position = new Vector2(0,0);
		target = new Vector2(0,0);
		zoom = new Vector2();
		targetZoom = new Vector2();
		zoom.x = (MAX_ZOOM_IN + MAX_ZOOM_OUT) * .5f;
		targetZoom.x = MAX_ZOOM_OUT * .5f; 			// init zoom
	}

	public void update(float deltaTime) {
		position.lerp(target, FOLLOW_SPEED); 
		
		if (targetZoom.x != zoom.x) {
			zoom.lerp(targetZoom, ZOOM_SPEED);
		}
	}

	public void applyTo(OrthographicCamera camera) {
		camera.position.set(position.x, position.y, 0);
		camera.zoom = zoom.x;
		camera.update();
	}

	public boolean hasTarget() {
		return target != null;
	}

	public boolean hasTarget(Vector2 target) {
		return hasTarget() && this.target.equals(target);
	}

	public void setTarget(Vector2 target) {
		this.target = target;
	}

	public void setPosition(float x, float y) {
		this.position.set(x, y);
	}

	public Vector2 getPosition() {
		return position;
	}

	public void addZoom(float amount) {
		targetZoom.x = MathUtils.clamp(targetZoom.x + amount, MAX_ZOOM_IN, MAX_ZOOM_OUT);
	}

	public float getZoom() {
		return zoom.x;
	}

	public void setZoom(float zoom) {
		this.zoom.x = zoom;
	}
	
	public void setTargetZoom(float zoom) {
		this.targetZoom.x = zoom;
	}

	public void setPosition(Vector2 position) {
		this.position = position;
	}
	

	public float getMinZoom() {
		return MAX_ZOOM_IN;
	}
	


	public float getMaxZoom() {
		return MAX_ZOOM_OUT;
	}

	public Body getFollowBody() {
		return followBody;
	}

	public void setFollowBody(Body followBody) {
		this.followBody = followBody;
	}
	
	
}
