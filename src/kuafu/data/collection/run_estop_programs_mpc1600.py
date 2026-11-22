#!/usr/bin/env python3
"""Enable the audited MPC1600 collector for the two Kinova E-stop tasks."""

from __future__ import annotations

import run_first2_new_programs_mpc1600 as campaign


campaign.DEFAULT_MANIFEST = (
    campaign.REPO_ROOT
    / "assets/configs"
    / "campaigns"
    / "mpc_estop_first_two_programs_1600.json"
)


campaign.EXPECTED_TASKS.update(
    {
        "estop_open": ("Kinova-Table-Gen3-EStop-Open", "kinova"),
        "estop_close": ("Kinova-Table-Gen3-EStop-Close", "kinova"),
    }
)


if __name__ == "__main__":
    campaign.main()
