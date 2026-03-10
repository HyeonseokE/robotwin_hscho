#!/bin/bash
cd /home/csi/RoboTwin/single_arm

TASKS=(
  adjust_bottle
  beat_block_hammer
  blocks_ranking_rgb
  blocks_ranking_size
  click_alarmclock
  click_bell
  move_can_pot
  move_pillbottle_pad
  move_playingcard_away
  move_stapler_pad
  open_laptop
  open_microwave
  place_a2b_left
  place_a2b_right
  place_bread_basket
  place_can_basket
  place_container_plate
  place_empty_cup
  place_fan
  place_mouse_pad
  place_object_basket
  place_object_scale
  place_object_stand
  place_phone_stand
  place_shoe
  press_stapler
  rotate_qrcode
  shake_bottle
  shake_bottle_horizontally
  stack_blocks_two
  stack_blocks_three
  stack_bowls_two
  stack_bowls_three
  stamp_seal
  turn_switch
)

CONFIG="demo_ur5_single_test"
PASS=0
FAIL=0
FAILED_TASKS=""

for task in "${TASKS[@]}"; do
  echo "=========================================="
  echo "Running: $task"
  echo "=========================================="
  rm -rf "data/${task}"

  timeout 300 python script/collect_data.py "$task" "$CONFIG" 2>&1
  EXIT_CODE=$?

  if [ $EXIT_CODE -eq 0 ] && [ -f "data/${task}/${CONFIG}/data/episode0.hdf5" ]; then
    echo ">>> PASS: $task"
    PASS=$((PASS + 1))
  else
    echo ">>> FAIL: $task (exit code: $EXIT_CODE)"
    FAIL=$((FAIL + 1))
    FAILED_TASKS="$FAILED_TASKS $task"
  fi
  echo ""
done

echo "=========================================="
echo "RESULTS: $PASS passed, $FAIL failed out of ${#TASKS[@]}"
if [ -n "$FAILED_TASKS" ]; then
  echo "FAILED TASKS:$FAILED_TASKS"
fi
echo "=========================================="
