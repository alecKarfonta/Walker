Description 
	Here is an application to experiment with Q-Learning to learn a control policy for arbtrary robots in a physics simulation. The main purpose of this program is to be able at runtime to adjust all of the various hyper-parameters around the search policy of Q-Learning, such as randomness and learning rate. Further there is an evolutionary component to speed up the search of Q agents and compare the performance of different combinations of parameters.


Project Details
	The project is written in Java and depends heavily on Libgdx and Box2d. Libgdx is a game development framework meant for building cross-platform games. Box2d is a popular physics engine. Like any Libgdx application there are several different projects. The root project simple called Walker contains metadata for the application. Walker-Core contains the logic of the application. Walker-Android is the Android launcher but also holds the resources for all platforms in the assets folder. Walker-Desktop is the launcher for PCs (Mac, Linux), and should have a link to the assets folder in the Android project. This is the project you will want to export for an exutable jar.


Import
	I would recommend using Eclipse. This is a Gradle project. So import as a Gradle project, go to the root folder with all of the projects and hit 'build'. Then select the projects you want to import, you will need at least the meta project, -Core, -Android, and -Desktop. See the Libgdx getting started guide for more information. 

Documentation (or lack there of)
	You might notice there is not much documentation. This is mostly experiemtal code that is contiually being replaced so I did not take the time to write proper class and method descriptions. The comments are also sparse and often wrong. I did try to make very verbose variable names. Also I use simple, consise lines that only do one thing at a time. 

Thanks for checking this out. I would appreciate any suggestions or contributions. Have fun.



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
