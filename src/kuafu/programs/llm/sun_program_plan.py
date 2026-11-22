from typing import Any, List, Dict, Tuple, Union, Set, FrozenSet


class SUNProgramPlan:
    """Track a SUN Program's task plan, atomic actions, and completed history.

    Completed tasks retain their atomic actions so a simulation reset can replay
    the accepted prefix before work resumes on the remaining tasks.
    """

    def __init__(self):

        self._reasoning = ""
        self._diagnosis = ""
        self._tasks = []  # high-level subtasks
        self._completed_tasks = []  # completed high-level subtasks
        self._completed_task_atomic_actions = []
        
        # Atomic actions for the current high-level subtask.
        self._current_atomic_actions = []
        self._atomic_actions_reasoning = ""
        # Track last completed atomic actions
        self._last_completed_atomic_actions = []


    def update_tasks(self, task_list: List[str] = []) -> None:
        self._tasks = task_list
        
    def update_atomic_actions(self, action_list: List[str] = []) -> None:
        """Update the atomic actions for the current high-level subtask"""
        self._current_atomic_actions = action_list

    def pop_task(self) -> None:
        if self._tasks:
            completed_task = self._tasks.pop(0)
            self._completed_tasks.append(completed_task)
            self._completed_task_atomic_actions.append(
                {
                    "task": completed_task,
                    "atomic_actions": self._current_atomic_actions.copy(),
                }
            )
            # Save current atomic actions before clearing them
            self._last_completed_atomic_actions = self._current_atomic_actions.copy()
            # Clear atomic actions when moving to next subtask
            self._current_atomic_actions = []

    @property
    def current_task(self) -> str:
        if len(self._tasks) > 0:
            return self._tasks[0]
        else:
            return ""

    @property
    def tasks(self) -> List[str]:
        return self._tasks
    
    @property
    def current_atomic_actions(self) -> List[str]:
        return self._current_atomic_actions
    
    @property
    def atomic_actions_reasoning(self) -> str:
        return self._atomic_actions_reasoning
    
    @atomic_actions_reasoning.setter
    def atomic_actions_reasoning(self, reasoning: str) -> None:
        self._atomic_actions_reasoning = reasoning

    @property
    def reasoning(self) -> str:
        return self._reasoning

    @property
    def diagnosis(self) -> str:
        return self._diagnosis
    
    @property
    def completed_tasks(self) -> List[str]:
        return self._completed_tasks

    @property
    def completed_task_atomic_actions(self) -> List[Dict[str, Any]]:
        return self._completed_task_atomic_actions
    
    @property
    def pending_tasks(self) -> List[str]:
        if len(self._tasks) > 1:
            return self._tasks[1:]
        else:
            return []
    
    @reasoning.setter
    def reasoning(self, reasoning: str) -> None:
        self._reasoning = reasoning

    @property
    def has_tasks(self) -> bool:
        return len(self._tasks) > 0
    
    @property
    def last_completed_atomic_actions(self) -> List[str]:
        """Get the last completed atomic actions from the previous subtask"""
        return self._last_completed_atomic_actions
