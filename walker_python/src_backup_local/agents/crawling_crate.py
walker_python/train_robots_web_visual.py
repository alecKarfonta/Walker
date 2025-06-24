    def get_reward(self, prev_x: float) -> float:
        x = self.body.position.x
        reward = (x - prev_x) * 10
        if abs(self.body.angle) > np.pi/2:
            reward -= 2.0
        return reward

    def take_action(self, action):
        self.apply_action(action)

    def update(self, delta_time):
        pass

    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information about the robot's state."""
        return {
            if hasattr(agent, 'q_table') and agent.q_table is not None:
                action = agent.get_action(state)
            else:
                # Small random torques
                action = (
                    np.random.uniform(-5.0, 5.0),
                    np.random.uniform(-5.0, 5.0)
                )
            
            # Apply action
            agent.apply_action(action)
            
        # Update episode counter
        self.episode += 1
        # End if agent falls or flips
        if agent.body.position.y < -2 or abs(agent.body.angle) > np.pi/2:
            break
            
        # Final fitness: forward progress
        final_fitness = agent.body.position.x - agent.position[0]
        return max(0, final_fitness)
        
    def run_training_step(self):
        """Run one step of training for all agents."""
        # ... existing code ...
            
            # Apply action
            agent.apply_action(action)
            
        # Update episode counter
        self.episode += 1
        # ... existing code ... 