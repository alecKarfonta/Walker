Interface Quick Guide

	Buttons
		Pause	:	Pause learning of selected robot
		Reset	:	Reset the Q-Table, memory, of selected robot
		Goal 	:	Cycle the selected robot's goal
		Add Crawling Crate	: 	Add a robot of the type with a single arm with two joints
		Add Legged Crate	: 	Add a robot of the type with two arms each with two joints
		Next	:	Select the next robot
		Previous	:	Select the previous robot
		Send Home	:	Send all crawling crates back to x = 0, they stop learning until they reach home
		Search Settings	:	Opens the window menu to control search settings for the world and the selected robot
		Save Settings	:	Saves the world settings from the search settings menu as the default
		
	Search Settings Menu	:	This menu contains controls for tweaking the world and the search policy of the selected robot. 
		E			:	The probability the robot will chose a random action
		Timestep	:	The timestep of the physics world, setting this value too high will cause peculiar things to happen 
		Alpha		:	The learning rate of the selected robot
		- Note: These next four values are used to control the limits on the adaptive learning rate and randomness 
		Min Randomness	:	The minimum probability that the selected robot will chose a random action
		Max Randomness	:	The maximum probability that the selected robot will chose a random action
		Min Learning Rate	: The minimum learning rate of the selected robot
		Max Learning Rate	: The maximum learning rate of the selected robot
		Arm	Speed	:	The speed the selected robot moves it's arm
		Wrist Speed	:	The speed the selected robot moves it's wrist
		isLearning	:	Controls whether the robot should perform Q-Updates or instead allow user control
		
	Zoom	:	With scroll wheel
	Robot	:	Each robot has several controls
		Mouse
			Click	:	Click a robot to select it. A selected robot has the camera centered on it, can have it's search policy altered in the Search Settings
						Menu, and can also be manually controlled by the user. 
			Drag	:	Click and drag a robot to move/throw it around
	Crawling Crate
		Down	:	Set arm joint motor to negative
		Up		:	Set arm joint motor to positive
		Left	:	Set wrist joint motor to negative
		Right	:	Set wrist joint motor to positive 
		Space	: 	Send just the selected robot home
		H		:	Toggle whether motors are held or released
		L		:	Toggle whether the robot is learning
