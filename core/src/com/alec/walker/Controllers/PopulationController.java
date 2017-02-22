package com.alec.walker.Controllers;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

import org.joda.time.Duration;

import com.alec.walker.Models.BasicAgent;
import com.alec.walker.Models.BasicPlayer;
import com.alec.walker.Models.CrawlingCrate;
import com.alec.walker.Models.Player;
import com.alec.walker.Views.Play;

public class PopulationController {
	// private ArrayList<BasicAgent> agents = new ArrayList<BasicAgent>();
	private Play					play;
	private WorldController			world;
	private Duration				age;

	public Player					selectedPlayer;

	private ArrayList<String>				first_names;
	private ArrayList<String>				last_names;

	public ArrayList<CrawlingCrate>	allPlayers	= new ArrayList<CrawlingCrate>();

	public PopulationController(Play play) {
		this.play = play;
		this.world = play.world;
		this.age = new Duration(0);

		// Load the list of names
		String csvFile = "names.csv";
		BufferedReader br = null;
		String line = "";
		String cvsSplitBy = ",";
		first_names = new ArrayList<String>();
		try {
			br = new BufferedReader(new FileReader(csvFile));
			while ((line = br.readLine()) != null) {
				first_names.add(line);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
		last_names = new ArrayList<String>();
		csvFile = "last_names.csv";
		try {
			br = new BufferedReader(new FileReader(csvFile));
			while ((line = br.readLine()) != null) {
				last_names.add(line);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}
	
	public void rank() {
		// Rank population
		Collections.sort(allPlayers, new Comparator<CrawlingCrate>() {
		    @Override
		    public int compare(CrawlingCrate o1, CrawlingCrate o2) {
		    	float x1 = o1.body.getPosition().x;
		    	float x2 = o2.body.getPosition().x;
		    	
		    	// Ascending
		    	if (x1 < x2) {
		    		return 1;
		    	} else if (x1 > x2) {
		    		return -1;
		    	}
		    	// Descending
//		    	if (x1 > x2) {
//		    		return 1;
//		    	} else if (x1 < x2) {
//		    		return -1;
//		    	}
		    	
		        return 0;
		    }
		});
		// Reindex
		for (int index = 0; index < allPlayers.size(); index++) {
			((CrawlingCrate)(allPlayers.get(index))).rank = index;
		}
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

	public CrawlingCrate makeCrawlingCrate() {

		CrawlingCrate crate = new CrawlingCrate(this.play);
		crate.init(world.world,
				(0 * (allPlayers.size() + 1)), world.groundHeight + 25);
		
		// Select a first last name randomly
		crate.name = first_names.get(new Random().nextInt(first_names.size())) + " " + last_names.get(new Random().nextInt(last_names.size()));

		allPlayers.add(crate);

		selectedPlayer = crate;

		return (CrawlingCrate) selectedPlayer;
	}

	public void cloneCrawlingCrate() {
		CrawlingCrate crate = ((CrawlingCrate) selectedPlayer).clone(world.world);

		// Copy name
		crate.name = ((CrawlingCrate) selectedPlayer).name;

		
		allPlayers.add(crate);
	}

	public CrawlingCrate spawnCrawlingCrate(CrawlingCrate parent) {
		System.out.println("PopulationControler.spawnCrawlingCrate(" + parent.name + ")");
		CrawlingCrate crate = parent.spawn(world.world);
		
		// Same last name as parent
		String lastName = parent.name.split(" ")[1];
		// Generate a new first name
		String firstName = first_names.get(new Random().nextInt(first_names.size()));
		crate.name = firstName + " " + lastName;
		System.out.println("Spawned " + crate.name + " from " + parent.name);
		
		crate.finishLine = parent.finishLine;
		
		allPlayers.add(crate);
		return crate;

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
