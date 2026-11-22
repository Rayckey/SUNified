import abc
from pathlib import Path, PosixPath
from typing import Any, Dict
import logging
from kuafu.programs.llm.io_util import load_txt, save_npz, load_npz, load_json
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseAgent:
    def __init__(self, program_runtime, agent_cfg, *args, **kwargs):
        self._load_path = Path(agent_cfg.load_path) if agent_cfg.load_path is not None else None

        self._program_runtime = program_runtime
        self._controller_api_wrapper = program_runtime.controller_api_wrapper


        self._goal_text = agent_cfg.task
        self._residual_types = []
        self._residual_literals = []
        self._residual_types_txt = []
        self._residual_literals_txt = []

        self._is_train = False

        self._trail_started = False
        self._num_steps = 0

    def __str__(self):
        return self.__class__.__name__

    def step(
        self
    ):
        """
        Advance the agent's step counter.
        """

        self._num_steps += 1

    @abc.abstractmethod
    def act(self, observation: Dict[str, Any]):
        """
        Predict action.
        """
        raise NotImplementedError("Override me!")

    @property
    def program_state(self):
        """
        Current agent program.
        """
        pass

    @program_state.setter
    def program_state(self, loaded_program_state: Dict[str, Any]):
        """
        Set agent program.
        """
        pass

    def save_program(self, save_dir: PosixPath):
        """
        Save current agent program.
        """
        if self.program_state is not None:
            logger.info(f"Save agent program to {save_dir}.")
            save_npz(self.program_state, save_dir / "program_state.npz")

    def load_program(self):
        """
        Load saved agent program.
        """
        if self._load_path is not None and self._load_path.exists():
            logger.info(f"Load agent program from {self._load_path}.")
            self.program_state = load_npz(self._load_path / "program_state.npz")
        else:
            logger.warning(f"Agent program path {self._load_path} does not exist! Skip loading.")

    def reset_trail(self, goal_spec: str, init_obs: Dict[str, Any], *args, **kwargs):
        """
        Reset agent to process new epoch.
        """
        pass

    def set_mode(self, is_train: bool):
        # switch mode only when the previous trail is finished
        assert not self._trail_started, "Cannot switch mode in the middle of a trail!"
        self._is_train = is_train

        logger.info(f"Agent mode set: is_train is {is_train}.")
