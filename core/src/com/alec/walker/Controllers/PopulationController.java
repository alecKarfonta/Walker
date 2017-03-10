package com.alec.walker.Controllers;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Random;

import com.alec.walker.Models.BasicPlayer;
import com.alec.walker.Models.CrawlingCrate;
import com.alec.walker.Models.Player;
import com.alec.walker.Models.StandingCrate;
import com.alec.walker.Views.Play;

public class PopulationController {
	// private ArrayList<BasicAgent> agents = new ArrayList<BasicAgent>();
	private Play					play;
	private WorldController			world;
	//private Duration				age;

	public Player					selectedPlayer;

	private ArrayList<String>				first_names;
	private ArrayList<String>				last_names;

	public ArrayList<StandingCrate>	allPlayers	= new ArrayList<StandingCrate>();

	public PopulationController(Play play) {
		this.play = play;
		this.world = play.world;
		//this.age = new Duration(0);

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
		Collections.sort(allPlayers, new Comparator<StandingCrate>() {
		    @Override
		    public int compare(StandingCrate o1, StandingCrate o2) {
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
			((StandingCrate)(allPlayers.get(index))).rank = index;
		}
	}

	public void init() {
		selectedPlayer = allPlayers.get(0);
	}

	public void update(float delta) {
//		age = age.plus((long) (delta * 1000));
	}

	public void removePlayer(BasicPlayer player) {
		System.out.println("PopulationController.removePlayer("+ player.name + ")");
		if (player instanceof StandingCrate) {
			System.out.println("Body count = " +  ((StandingCrate)player).bodies.size());
			world.destroyQueue.addAll( ((StandingCrate)player).bodies);
			System.out.println("Joint count = " +  ((StandingCrate)player).joints.size());
			world.destroyJointQueue.addAll( ((StandingCrate)player).joints);
			allPlayers.remove(player);
		}
	}

//	public String getAge() {
//		// Bot count
//		String worldAge = "";
//		if (age.getStandardHours() > 0) {
//			worldAge = age.getStandardHours() + "h";
//		} else if (age.getStandardMinutes() > 0) {
//			worldAge = age.getStandardMinutes() + "m";
//		} else {
//			worldAge = age.getStandardSeconds() + "s";
//		}
//		return worldAge;
//	}

	public StandingCrate makeStandingCrate() {

		StandingCrate crate = new StandingCrate(this.play);
		crate.init(world.world,
				(0 * (allPlayers.size() + 1)), world.groundHeight + 10 +  crate.height * 2);
		
		// Select a first last name randomly
		crate.name = first_names.get(new Random().nextInt(first_names.size())) + " " + last_names.get(new Random().nextInt(last_names.size()));

		allPlayers.add(crate);

		selectedPlayer = crate;

		return (StandingCrate) selectedPlayer;
	}

	public void cloneStandingCrate() {
		StandingCrate crate = ((StandingCrate) selectedPlayer).clone(world.world);

		// Copy name
		crate.name = ((StandingCrate) selectedPlayer).name;

		
		allPlayers.add(crate);
	}

	public StandingCrate spawnStandingCrate(StandingCrate parent) {
		System.out.println("PopulationControler.spawnStandingCrate(" + parent.name + ")");
		StandingCrate crate = parent.spawn(world.world);
		
		// Select a first last name randomly
		crate.name = first_names.get(new Random().nextInt(first_names.size())) + " " + parent.name.split(" ")[1];

		System.out.println("Spawned " + crate.name + ")");
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

//	public CrawlingCrate makeCrawlingCrate() {
//		CrawlingCrate crate = new CrawlingCrate(this.play);
//		crate.init(world.world,
//				(0 * (allPlayers.size() + 1)), world.groundHeight + 10 +  crate.height * 2, 1, 1);
//		
		// Select a first last name randomly
//		crate.name = first_names.get(new Random().nextInt(first_names.size())) + " " + last_names.get(new Random().nextInt(last_names.size()));
//
//		allPlayers.add(crate);
//
//		selectedPlayer = crate;
//
//		return (CrawlingCrate) selectedPlayer;
//	}
	
}
