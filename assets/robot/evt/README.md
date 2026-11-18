# EVT2 Simple model

The runtime source for training and native MuJoCo evaluation is `urdf/evt2_simple.xml`.
It is derived from the retained xSIM source `urdf/evt2.xml`, borrowing non-foot primitive collision
shapes from `urdf/evt2_mjlab.xml`. Rebuild with `python -m tools.build_evt_simple`.

Body frames, mass, local COM, inertia frames, joint limits, armature, passive damping, frictionloss,
motors, sensors and the ground are preserved from EVT2. The mjlab variant's damping values are not used.
All mesh collisions are disabled. Collision geometry consists of 36 capsules, 2 boxes and one plane.
Foot capsules retain the original positions and radii and subtract the radius from the cylinder half-length;
their total envelopes fit within the original cylinders (about 24 by 8.4 cm per foot).

Training removes the 29 disabled position actuators in memory and wraps the 29 existing XML motors.
Both training and sim2sim compute `Kp * (target - q) - Kd * qdot` every 0.005 seconds, with PD gains from
config.json and the motor/joint limits from XML. Policy targets update every 0.02 seconds.
Solver settings use mjlab defaults with only timestep overridden. Default sim2sim applies those same settings.
The XML retains only the actuator-disable option, not custom numerical solver settings.

`python -m tools.audit_evt_simple` checks exact EVT2 parameter preservation, foot bounds, and independently
fuses the URDF inertias with explicit fixed-joint transforms and the parallel-axis theorem. URDF comparison
allows only the original EVT2 decimal serialization precision. It does not use MuJoCo's URDF fixed-link fusion.

Training friction randomization scales the XML foot and ground coefficients together by 0.8–1.2.
COM randomization is centered on the correct EVT2 COM of waist_pitch_link; evaluation restores nominal values.
The old `mjcf/evt.xml` and its URDF conversion script are not runtime sources.

See [training and validation](../../../docs/evt2_training.md). Equal foot envelopes do not imply identical
contact distributions: validate the deployed robot collision model before deployment.
