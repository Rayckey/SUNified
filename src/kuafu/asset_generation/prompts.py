"""Load the editable asset reconstruction prompts."""
from kuafu.paths import PROMPT_ROOT

BUILD_INSTRUCTIONS = (PROMPT_ROOT / "asset_generation/build_instructions.txt").read_text()
REVIEW_INSTRUCTIONS = (PROMPT_ROOT / "asset_generation/review_instructions.txt").read_text()
