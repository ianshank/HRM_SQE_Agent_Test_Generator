"""
Puzzle environment for evaluation.

Simulates puzzle-solving environment with step-by-step execution.
TODO: Integrate with actual puzzle simulation logic.
"""

import torch
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PuzzleEnvironment:
    """
    Environment for simulating puzzle solving.
    
    Provides step-by-step interaction for evaluating puzzle-solving models.
    This is a template implementation that should be adapted to your
    specific puzzle domain.
    """
    
    def __init__(
        self,
        puzzle_id: int,
        max_steps: int = 1000,
        timeout: float = 60.0,
    ):
        """
        Initialize puzzle environment.
        
        Args:
            puzzle_id: ID of the puzzle to solve
            max_steps: Maximum number of steps allowed
            timeout: Timeout in seconds
        """
        self.puzzle_id = puzzle_id
        self.max_steps = max_steps
        self.timeout = timeout
        
        self.current_step = 0
        self.done = False
        self.solved = False
        
        self.state_history = []
        self.action_history = []
        
        logger.debug(f"Initialized PuzzleEnvironment for puzzle {puzzle_id}")
    
    def reset(self) -> torch.Tensor:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation
        """
        self.current_step = 0
        self.done = False
        self.solved = False
        self.state_history = []
        self.action_history = []
        
        initial_state = self._get_initial_state()
        self.state_history.append(initial_state)
        
        return initial_state
    
    def step(
        self,
        action: int
    ) -> Tuple[torch.Tensor, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        if self.done:
            raise RuntimeError("Environment is done. Call reset() first.")
        
        self.current_step += 1
        self.action_history.append(action)
        
        next_state = self._compute_next_state(action)
        self.state_history.append(next_state)
        
        reward = self._compute_reward(action)
        
        self.done = (
            self.current_step >= self.max_steps or
            self._check_solved()
        )
        
        if self._check_solved():
            self.solved = True
            reward += 10.0  # Bonus for solving
        
        info = {
            "step": self.current_step,
            "solved": self.solved,
            "puzzle_id": self.puzzle_id,
        }
        
        return next_state, reward, self.done, info
    
    def _get_initial_state(self) -> torch.Tensor:
        """
        Get initial state for the puzzle.
        
        Returns:
            Initial state tensor
            
        Note:
            This is a mock implementation. Replace with actual puzzle state.
        """
        return torch.randint(0, 12, (10,))
    
    def _compute_next_state(self, action: int) -> torch.Tensor:
        """
        Compute next state given current state and action.
        
        Args:
            action: Action taken
            
        Returns:
            Next state tensor
            
        Note:
            This is a mock implementation. Replace with actual state transition.
        """
        current_state = self.state_history[-1]
        
        next_state = current_state.clone()
        
        if action == 0:
            next_state = torch.roll(next_state, 1)
        elif action == 1:
            next_state = torch.flip(next_state, [0])
        
        return next_state
    
    def _compute_reward(self, action: int) -> float:
        """
        Compute reward for the action.
        
        Args:
            action: Action taken
            
        Returns:
            Reward value
            
        Note:
            This is a mock implementation. Replace with actual reward logic.
        """
        return -0.1  # Step penalty
    
    def _check_solved(self) -> bool:
        """
        Check if puzzle is solved.
        
        Returns:
            True if puzzle is solved
            
        Note:
            This is a mock implementation. Replace with actual solve check.
        """
        if len(self.state_history) < 2:
            return False
        
        current_state = self.state_history[-1]
        
        return torch.all(current_state == current_state.sort()[0]).item()
    
    def get_trajectory(self) -> Dict[str, Any]:
        """
        Get complete trajectory of the episode.
        
        Returns:
            Dictionary with states and actions
        """
        return {
            "puzzle_id": self.puzzle_id,
            "states": [s.tolist() for s in self.state_history],
            "actions": self.action_history,
            "num_steps": self.current_step,
            "solved": self.solved,
        }

