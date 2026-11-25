# SO-101 LeRobot setup and operator guide

This guide covers the Homebrain setup used for the wrist-only blue-cube-to-white-box
experiment. The Studio interface is in Russian. The instructions include the button
labels shown in the interface.

## 1. Set up the Jetson

The tested environment is Linux on an NVIDIA Jetson with Python 3.12, LeRobot
0.6.1, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, and MLflow 3.6.0.
Use a Jetson-compatible NVIDIA/PyTorch build matching the installed JetPack and
CUDA stack. The [requirements lock](../config/requirements.lock.txt) lists the
Homebrain environment. The SAM 3 entry points to a local checkout, and the CUDA
wheel suffixes require a compatible package source. A default PyPI install
alone does not reproduce this environment.

Follow the [clone commands in the README](../README.md#quick-start-record-your-first-demonstration).
From the checkout, create a private environment and install the matched PyTorch and
torchvision wheels. Then install the rest of the locked packages and the lab:

```bash
python3.12 -m venv .venv
# Install the Jetson-compatible torch and torchvision wheels here first.
awk '!/^(sam3 @ |torch==|torchvision==)/' config/requirements.lock.txt |
  .venv/bin/python -m pip install -r /dev/stdin
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

For CUDA training the last value should be `True`. The shipped systemd units,
`start_services.sh`, and the SAM 3 setup script use the Homebrain checkout path
`/home/jetsonnx/Projects/so101_ai_learning`. Use that path when reproducing the
Homebrain installation or adapt those paths for the new host.

You can go straight to [hardware setup](#2-connect-and-check-the-hardware)
to record your first demonstrations. Provision model assets under the service
user's normal caches before running annotation or starting offline training:

| Feature | Required local assets |
| --- | --- |
| ACT with pretrained vision backbone | Torchvision ResNet18 ImageNet V1 weights in the Torch hub cache. The full ACT policy is trained from demonstrations. |
| SmolVLA | `lerobot/smolvla_base` and `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` in `~/.cache/huggingface/hub`. Both snapshots are used. |
| SAM 3 | Install the pinned SAM 3 source with [fetch_sam_source.sh](../scripts/fetch_sam_source.sh) at the Homebrain path. Obtain authorized access to `facebook/sam3` and cache `sam3.pt` under the same Hugging Face hub cache. |

For example, after signing in to Hugging Face when required, prefetch the models
you intend to use:

```bash
.venv/bin/python -m huggingface_hub.cli.hf download lerobot/smolvla_base
.venv/bin/python -m huggingface_hub.cli.hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct
.venv/bin/python -m huggingface_hub.cli.hf download facebook/sam3 sam3.pt
.venv/bin/python -c 'from torchvision.models import ResNet18_Weights; ResNet18_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True)'
```

The baseline models use RGB and joints. SAM masks are optional for that input
mode. A missing SAM package or model can still appear in the overall model
preflight, which checks the whole installation.

## 2. Connect and check the hardware

1. Connect the powered SO-101 leader and follower and mount the USB wrist
   camera rigidly. Connect the overhead camera for dual-camera capture.
2. Verify that `/dev/so101-leader` and `/dev/so101-follower` resolve to the correct
   serial devices. On a new machine, use local udev rules to create stable
   aliases from each controller's USB identity. Ensure the service user has
   serial and video access. Use `ls -l /dev/serial/by-id /dev/v4l/by-id` to
   identify the devices. If your cameras have different USB IDs, update
   `DEFAULT_CAMERAS` in [hardware.py](../src/so101_lab/hardware.py) in your
   checkout to use your stable video paths. The verified recording
   setup is 640 × 480 at 30 FPS, YUYV for wrist and MJPG for overhead.
3. Use calibrations measured for the actual leader and follower. Homebrain's
   existing files in `config/calibration_snapshots/` are specific to its arms.
   For different arms, use the [LeRobot SO-101 calibration procedure](https://huggingface.co/docs/lerobot/so101) with
   IDs `so101_leader` and `so101_follower`. Keep the follower calibration in the
   LeRobot cache for the official rollout. Copy the corresponding files into
   the lab as `leader.json` and `follower.json`. Set their SHA-256 values in
   `manifest.json` to match those files, and update the provenance fields.
   The lab checks the snapshot and compares it with hardware when a session
   starts. Camera positioning is separate from motor calibration.
4. Ensure the selected devices are free before preview, recording, or rollout.
   Close any competing LeLab, Wristlab, or pick-and-place session via its own
   controls. Keep the workspace clear and the hardware STOP reachable.

For a different or newly assembled pair, run these only during an operator-led
calibration session with other controllers closed. Follow LeRobot's interactive
prompts for the corresponding arm:

```bash
.venv/bin/python -m lerobot.scripts.lerobot_calibrate \
  --teleop.type=so101_leader --teleop.port=/dev/so101-leader --teleop.id=so101_leader
.venv/bin/python -m lerobot.scripts.lerobot_calibrate \
  --robot.type=so101_follower --robot.port=/dev/so101-follower --robot.id=so101_follower
sha256sum config/calibration_snapshots/{leader,follower}.json
```

Run the last line after copying the measured calibration files. Use the resulting
hashes in the manifest. Homebrain's working arms already have their calibrations.

### Start Studio and MLflow

To start Studio from a clone in any directory, run this from its root:

```bash
mkdir -p runtime logs models data/mlflow-artifacts
PYTHONPATH="$PWD/src" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -m uvicorn so101_lab.app:app --host 0.0.0.0 --port 8020
```

Leave that terminal running. Open `http://<jetson-address>:8020` on a trusted
LAN or use `http://localhost:8020` from the robot host. Opening Studio gives
you the existing recording interface and the Leader → Follower controls.
MLflow is optional for the first recording. To use it, run this in another
terminal from the same directory:

```bash
.venv/bin/python -m mlflow server --host 0.0.0.0 --port 5020 --workers 1 \
  --backend-store-uri "sqlite:///$PWD/runtime/mlflow.db" \
  --artifacts-destination "$PWD/data/mlflow-artifacts" \
  --allowed-hosts "localhost:*,127.0.0.1:*,$(hostname):*"
```

To open MLflow from another computer, include the Jetson IP or DNS name in
`--allowed-hosts`. Failed deliveries are spooled under `runtime/mlflow-spool/`.

On the existing Homebrain installation with the user units already installed:

```bash
./scripts/start_services.sh
curl -fsS http://127.0.0.1:8020/api/status
curl -fsS http://127.0.0.1:5020/health
```

For a new installation at the expected Homebrain path, review the addresses in
the [unit files](../config/systemd/) and then install them:

```bash
mkdir -p ~/.config/systemd/user runtime logs data/mlflow-artifacts models
install -m 644 config/systemd/so101-ai-learning-* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now so101-ai-learning-ui.service \
  so101-ai-learning-mlflow.service so101-ai-learning-mlflow-replay.timer
```

Open `http://<jetson-address>:8020` for Studio and port `5020` for MLflow.
For Homebrain the LAN address was `192.168.0.168`. The application serves live
controls over HTTP, so only expose it within a trusted network.

## 3. Record demonstrations

1. Open **Record**, select **Wrist camera only** (`Только камера на руке`), and
   connect the cameras (`Подключить камеры`). Confirm the wrist preview is live.
   Wrist-only stores wrist RGB, measured joints, and the actions sent to the
   follower. The generic preflight inventories both cameras, so check the
   actual selected mode if an unused overhead camera appears missing.
2. Run **Preflight** (`Проверить перед записью`). For a new rig, do a separate
   supervised teleoperation check (`Включить телеоперацию`) before recording.
   Verify how the follower tracks every joint and the gripper, then end
   teleoperation using `STOP рук`.
3. Select **New dataset** (`Новый датасет`). Enter a name, the task, and the
   total target count. For this experiment use one consistent instruction,
   such as `Pick up the blue cube, place it in the white box, release it, and
   return the empty gripper to the starting view.` Preserve the same meaning
   across demonstrations and use the recorded instruction for SmolVLA. The
   original Homebrain dataset uses `Возьми один синий кубик, положи его в белую
   коробку, отпусти и верни пустой захват в исходный обзор`.
4. Record one pilot episode. Inspect its playback before collecting the
   remaining episodes. Start a session with `Начать запись`. The preparation
   countdown gives time to set the scene. **Space** starts early or saves the
   active episode. `Сохранить эпизод` also saves. **R** or `Перезаписать`
   archives the attempt and prepares a replacement. **Esc** or
   `Остановить` ends the session. If the current good attempt is still
   recording, save it before stopping. An interrupted attempt goes to
   quarantine and is excluded from training.
5. To pause collection, end the session after saving the current good episode.
   In **Datasets** (`Датасеты`), select **Continue** (`Продолжить · n / total`)
   on the same dataset card. The saved count and original total carry
   over. You can divide 50 episodes into several sessions without creating
   separate datasets.

### What a good episode looks like

- Keep the camera mount, box, and lighting consistent for the first task. At
  the initial view the wrist camera should clearly show the cube. The box can
  appear once the arm turns. While placing the cube, bring the box and release
  area into view.
- Start from a repeatable arm pose. Vary cube positions across the planned
  pickup area and record multiple clean examples at each part of the area.
  Include the small orientation and framing changes you want the policy to
  handle. Avoid collecting all examples from a single spot.
- Demonstrate a complete sequence at a steady speed: approach, grasp, lift,
  carry, lower into the box, open, clear the rim, and come back to the
  starting view. Give the camera a clear view of the completed release before
  ending the episode. Reset the scene during preparation for the next one.
- Review the wrist playback, especially the first view, closure, approach to
  the box, and release. Replace attempts with missed grasps, drops before
  release, bad visibility, USB faults, or an accidental stop. Verify that
  examples with clearly different cube positions still contain a clean finish.
- Use several recording sessions so validation and test can hold out entire
  sessions. Aim for coverage across train and held-out sessions. Fifty was the
  count used here. Successful placement and repeatability are still evaluated
  on the robot.

## 4. Review, validate, and export

In **Datasets**, use `Смотреть эпизоды` to review clips, then run `Проверить
данные` and `Завершить и сохранить LeRobot`. The latter exports the valid
saved demonstrations to LeRobot v3 format. The Studio trainer can use the
validated raw dataset. The export is useful for independent LeRobot tools
and dataset transfer. The CLI runs the same CPU backend:

```bash
.venv/bin/python scripts/export_lerobot.py data/ds_YOUR_DATASET --validate-only
.venv/bin/python scripts/export_lerobot.py data/ds_YOUR_DATASET
```

SAM 3 can annotate the saved frames in the background. Automatic full
annotation is scheduled after the saved episode target is reached. Use the
annotation job's completed/total frame counts and review ambiguous masks if
you intend to choose **SAM object features**. The baseline uses RGB and
joints, so it can train independently of annotation results. RGB masks
provide image locations and categories; metric object poses require separate
geometry.

## 5. Train ACT and SmolVLA

Open **Training** (`Обучение`). Choose the verified dataset, **Wrist-only**,
**RGB + joints**, and **CUDA**. For evaluation, use whole recording sessions
for train, validation, and test. Choose **Check splits** (`Проверить выборки`)
and verify the actual session and frame counts before selecting `Начать
обучение`. The earlier evaluation used 36/9/5 episodes, with 14,273/3,049/1,638
frames respectively. Keep the test split out of checkpoint selection.

The following is the later Homebrain configuration, using all 50 selected
episodes and image augmentation. Set **Validation = 0%** and **Test = 0%**
only when deliberately reproducing that final fit. Its results are training
losses, and the saved runs with held-out sessions remain the offline reference.

| Setting | ACT | SmolVLA |
| --- | ---: | ---: |
| Optimizer updates | 30,000 | 15,000 |
| Batch / gradient accumulation | 4 / 1 | 1 / 4 |
| Learning rate | 0.0001 | 0.0001 |
| Optimizer / schedule | AdamW / cosine | AdamW / cosine |
| Warmup updates | 1,000 | 750 |
| AMP | BF16 | BF16 |
| Seed / data workers | 42 / 2 | 42 / 0 |
| Image size / augmentation | 256 / on | 256 / on |
| Predicted action chunk | 50 | 50 |
| Actions per prediction for deployment | 50, temporal ensemble off | 10 |
| Save every / checkpoints retained | 5,000 / 6 | 3,000 / 4 |

The ACT run used a ResNet18 pretrained vision backbone (backbone LR
`0.00001`, KL weight `10`). SmolVLA started from `lerobot/smolvla_base` with
10 denoising steps. ACT uses images and state. SmolVLA also uses the task text.
The historical ACT run stored an experimental one-action execution option.
The final checkpoint was corrected to the 50-action deployment profile
without changing its weights. For new ACT runs use the current 50-action
default with temporal ensembling off.

Watch progress under **Training jobs** and in MLflow on port `5020`. The
worker continues with the browser closed. Jobs sharing the GPU wait for one
another, and metrics can spool locally if MLflow is temporarily unavailable.
For resume use the whole checkpoint directory and choose a final update
count greater than the saved step. Preserve the dataset, split, and model
input configuration.

## 6. Evaluate on the physical follower

1. Prepare the same task setup and a starting view comparable to the
   demonstrations. Clear the motion area. Stay by the hardware STOP.
2. Open **Run on robot** (`Запуск на роботе`), select the desired checkpoint,
   use the demonstration instruction, and run `Проверить модель и задержку`.
   This loads the checkpoint on an archived observation without motion.
3. Check the preview and start the policy explicitly from `Запустить
   политику`. Use the UI STOP for unexpected behavior and hardware STOP when
   needed. Observe whether the robot grasps, carries, releases inside the
   box, and clears the box.
4. Record the outcome by phase and repeat with different cube positions from
   the demonstrated area. A process exit code of zero records completion of
   the software run. Use direct observation to score the physical task.

The Homebrain final-fit ACT run is
`aed3d3bcab144b7482693cc7167a9c45/checkpoints/step_00030000`. Its training
loss was about 0.1744. After the execution correction the operator confirmed
that the robot picked up the blue cube and got it into the white box in one
physical trial. Placement precision and repeatability still need to be
measured. The final SmolVLA run is
`118a1aee7f6242b9aebbb0a68e9db34b/checkpoints/step_00015000`. Its training loss
was about 0.1242. A successful physical grasp from that checkpoint is unverified.

## Artifacts and troubleshooting

| Location | Contents |
| --- | --- |
| `data/<dataset-id>/episodes/` and `quarantine/` | Saved demonstrations and rejected attempts |
| `data/<dataset-id>/exports/` | Finalized LeRobot data and source provenance |
| `models/<job-id>/checkpoints/` | Complete checkpoints including config, weights, processors, and resume state |
| `runtime/jobs/` and `logs/` | Job status and rollout traces |
| `runtime/mlflow.db`, `runtime/mlflow-spool/`, and `data/mlflow-artifacts/` | Local metrics database, pending telemetry, and MLflow artifacts |

- **A device is busy:** use the named owner's interface to end its session,
  then check preflight again.
- **Calibration does not match:** verify you mapped the two physical arms
  to the right device aliases and that the cached and copied calibrations
  belong to those same arms.
- **Training waits for resources:** inspect the active job and available Jetson
  memory. The preflight estimates checkpoint disk usage. The 2 GB recording
  readiness floor is not the space required for a full dataset or train run.
- **ACT stalls before closing:** verify the checkpoint uses the saved
  centered letterbox processor at its trained image size and `n_action_steps=50`
  with no temporal ensemble for the 50-action profile. For older lab
  checkpoints, [fix_act_preprocessor.py](../scripts/fix_act_preprocessor.py)
  repairs the processor/config while retaining backups. Recheck loading
  after any change.
- **A cycle completes without a successful grasp or release:** inspect the
  saved wrist frames and joint/action trace together to identify the first
  phase that diverged. The rollout diagnostics are in
  `logs/lerobot-rollout-*.trace.jsonl`.
