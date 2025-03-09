Manipulator models taken from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) at commit [d3b4507](https://github.com/google-deepmind/mujoco_menagerie/commit/d3b4507788d70ccc150cf7a5090e99167f6159b1).

## Model modifications:

[franka_emika_panda/panda.xml](./franka_emika_panda/panda.xml)
- Added a end effector site named `ee_site`

[universal_robots_ur5e/ur5e.xml](./universal_robots_ur5e/ur5e.xml)
- Set joint limits to `+-pi` instead of `+-2pi` (these joint limits give better results with bi-directional RRT)
- Made `attachment_site` easily visible in the MuJoCo viewer by increasing the site size, changing the site color to red, and changing the site group to 1
