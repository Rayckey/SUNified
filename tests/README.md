# Validation

Follow the [development checks](../docs/DEVELOPMENT.md#validation).
`agents/` covers tool contracts, MCP and persistent handoffs; `asset_generation/`
covers geometry and articulation; `stage_dp3/` covers diffusion data and policies.
Other files cover program compilation, controllers, collection and learning.
`test_task_catalog.py` checks registrations, saved-program families, scene sources
and bundled assets for the public task catalog, with explicit coverage exclusions
for removed examples.
