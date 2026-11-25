# SO-101 LeRobot Imitation Learning

Record teleoperated SO-101 demonstrations, validate and export them as LeRobot
datasets, train ACT and SmolVLA on a Jetson, track runs in MLflow, and evaluate
checkpoints on the physical follower. The browser manages the workflow. Models,
data, and robot control run locally on Homebrain.

## Quick start: record your first demonstration

Clone `main` on the Linux machine connected to the SO-101 arms. Use your Mac
or another computer as the browser.

```bash
git clone https://github.com/vladromenko/so101-lerobot-imitation-learning.git so101_ai_learning
cd so101_ai_learning
```

1. [Install the Python and LeRobot environment](docs/OPERATOR_GUIDE.md#1-set-up-the-jetson).
   You can record before downloading SmolVLA or SAM 3 weights.
2. [Connect and calibrate the arms and wrist camera](docs/OPERATOR_GUIDE.md#2-connect-and-check-the-hardware).
   Match your USB devices and use the calibration for your own arms.
3. [Start Studio from the checkout](docs/OPERATOR_GUIDE.md#start-studio-and-mlflow)
   and open `http://<jetson-address>:8020`.
4. In **Record** (`Запись`), choose **Wrist camera only** (`Только камера на руке`)
   and connect the camera. Run preflight and check Leader → Follower
   teleoperation (`Включить телеоперацию`).
5. Enter a dataset name and task, set an episode target, then click
   `Начать запись`. Save a good attempt with `Сохранить эпизод` and review it
   under **Datasets** (`Датасеты` → `Смотреть эпизоды`).

The [English operator guide](docs/OPERATOR_GUIDE.md) explains what makes a
good demonstration, how to resume the same dataset, and how to export, train,
and evaluate an ACT or SmolVLA policy.

## What was tested

- Recorded 50 wrist-only demonstrations (18,960 frames) of picking up a blue
  cube and placing it in a white box. Demonstrations use calibrated
  leader-to-follower teleoperation and store RGB, measured joints, and the
  commands sent to the follower.
- Trained ACT and SmolVLA with a held-out session split. Both were later
  retrained on all 50 episodes with image augmentation.
- The final ACT checkpoint reached 30,000 updates. With its original weights,
  matched image preprocessing, and the restored LeRobot execution profile of
  50 actions per prediction, it physically picked up a cube and carried it
  almost to the box. One grasp was confirmed by the operator. Box placement,
  release, and repeatability remain to be verified.
- The final SmolVLA checkpoint reached 15,000 updates. This final fit has a
  training result; a successful physical grasp from it is unverified.

## Project layout

| Path | Purpose |
| --- | --- |
| [static/](static/) | SO-101 Studio browser interface |
| [teleoperation.py](src/so101_lab/teleoperation.py), [hardware.py](src/so101_lab/hardware.py) | Calibrated Leader → Follower control and hardware I/O |
| [recording.py](src/so101_lab/recording.py) | Recording and saving episodes, resuming sessions, and quarantining interrupted attempts |
| [training.py](src/so101_lab/training.py), [inference.py](src/so101_lab/inference.py) | ACT and SmolVLA training, checkpoint checks, and policy rollout |
| [src/so101_lab/lerobot_plugin/](src/so101_lab/lerobot_plugin/) | LeRobot image preprocessing and rollout diagnostics |
| [config/](config/) | Environment snapshot, hardware calibration snapshots, and service definitions |
| [scripts/](scripts/) | Dataset export, service startup, and checkpoint maintenance |

Studio uses port `8020` and MLflow uses port `5020`. Live controls should stay
on the local machine or a trusted LAN/VPN. The services start with robot
movement disabled until an explicit operator action.

Raw demonstrations, checkpoints, logs, and the MLflow database are local
artifacts. A usable checkpoint transfer includes its entire directory, with
weights, configuration, and processors. See the [operator guide](docs/OPERATOR_GUIDE.md#artifacts-and-troubleshooting)
for local paths.

The geometry-based planner and its original 3D and kinematics viewer live on
the [Wristlab branch](https://github.com/vladromenko/so101-lerobot-imitation-learning/tree/codex/so101-wristlab).
[Historical Homebrain notes in Russian](README_RU.md) preserve earlier
implementation details and diagnostics.
