package com.alec.walker.Controllers;

import java.util.ArrayList;

import org.joda.time.Duration;

import com.alec.walker.Models.BasicAgent;
import com.alec.walker.Models.BasicPlayer;
import com.alec.walker.Models.CrawlingCrate;
import com.alec.walker.Models.Player;

public class PopulationController {
	// private ArrayList<BasicAgent> agents = new ArrayList<BasicAgent>();
	private WorldController			world;
	private Duration				age;

	private Player					selectedPlayer;

	public ArrayList<BasicPlayer>	allPlayers	= new ArrayList<BasicPlayer>();

	public PopulationController(WorldController world){
		this.world = world;
		this.age = new Duration(0);
	}
	
	public void init() {

		selectedPlayer = allPlayers.get(0);
	}

	public void update(float delta) {

		age = age.plus((long) (delta * 1000));
	}

	public void removePlayer(BasicPlayer player) {
		world.destroyQueue.addAll(player.getBodies());
		allPlayers.remove(player);
	}

	public String getAge() {
		// Bot count
		String worldAge = "";
		if (age.getStandardHours() > 0) {
			worldAge = age.getStandardHours() + "h";
		} else if (age.getStandardMinutes() > 0) {
			worldAge = age.getStandardMinutes() + "m";
		} else {
			worldAge = age.getStandardSeconds() + "s";
		}
		return worldAge;
	}
	
	public BasicAgent makeCrawlingCrate() {

		CrawlingCrate crate = new CrawlingCrate();
		crate.init(world.world,
				(0 * (allPlayers.size() + 1)), world.groundHeight + 10, // (x,y)
				8, 4);

		allPlayers.add(crate);

		selectedPlayer = crate;
		
		return  (BasicAgent) selectedPlayer;
	}
	

	public void cloneCrawlingCrate() {
		CrawlingCrate crate = ((CrawlingCrate)selectedPlayer).clone(world.world);

		allPlayers.add(crate);
	}

	public void spawnCrawlingCrate() {
		CrawlingCrate crate = ((CrawlingCrate)selectedPlayer).spawn(world.world);
		
		allPlayers.add(crate);
	
	}
	public void selectPlayer(Player player) {
		this.selectedPlayer = player;
		
	}
	
	public Player previousPlayer() {

		int playerIndex = allPlayers.indexOf(selectedPlayer);
		if (playerIndex > 1) {
			playerIndex = playerIndex - 1;
		} else {
			playerIndex = allPlayers.size() - 1;
		}
		selectedPlayer = allPlayers.get(playerIndex);
		
		return selectedPlayer;
	}
	
	public Player nextPlayer() {

		int playerIndex = allPlayers.indexOf(selectedPlayer);

		if (playerIndex > allPlayers.size() - 2) {
			playerIndex = 0;
		} else {
			playerIndex = playerIndex + 1;
		}
		selectedPlayer = allPlayers.get(playerIndex);
		
		return selectedPlayer;
	}

	
}
