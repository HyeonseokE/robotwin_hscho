# All variable names for task information must be in uppercase.

# Template of Task Information:
"""
TASK_NAME = {
    "task_name": "task_name",                # Name of the task
    "task_description": "...",               # Detailed description of the task
    "current_code": '''
                class gpt_{task_name}({task_name}):
                    def play_once(self):
                        pass
                '''                          # Code template to be completed
    "actor_list": {                          # List of involved objects; can be a dictionary or a simple list
        "self.object1": {
            "name": "object1",               # Object name
            "description": "...",            # Description of the object
            "modelname": "model_name"        # Name of the 3D model representing the object
        },
        "self.object2": {
            "name": "object2",
            "description": "...",
            "modelname": "model_name"
        },
        # ... more objects
    },
    # Alternatively, the actor_list can be a simple list:
    # "actor_list": ["self.object1", "self.object2", ...],
    # To make Code Generation easier, the actor_list also includes some pose like target pose or middle pose, this is optional and dont have modelname.
}
"""

################## Known Tasks ##################
# These tasks are used to debug and iterate on prompt design.
# Prompt instructions have been specifically adjusted for them.

BEAT_BLOCK_HAMMER = {
    "task_name": "beat_block_hammer",
    "task_description":
    "Pick up the hammer and use it to beat the block on the table once. The hammer is placed at a fixed position on the table, \
                        but the block is generated randomly on the table. If the block's x coordinate (dim 0) is greater than 0, use the right arm to grasp the hammer, \
                        else use the left arm. To beat the block, you should place the hammer on the block's functional point \
                        (i.e., use the place_actor API to align the hammer's contact point with the block's functional point). \
                        Note: You don't need to Lift the hammer after beating the block, and you don't need to open the gripper or return the arm to origin position.",
    "current_code": """
                class gpt_beat_block_hammer(beat_block_hammer):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.hammer": {
            "name": "hammer",
            "description": "The hammer used to beat the block.",
            "modelname": "020_hammer"
        },
        "self.block": {
            "name": "block",
            "description": "The block that needs to be beaten by the hammer.",
            "modelname": "sapien-block1",
        }
    },
}

STACK_BLOCKS_TWO = {
    "task_name": "stack_blocks_two",
    "task_description":
    "Use the gripper to pick up block1 and move block 1 to the target position. Then pick up block 2 and place it on the block 1.\
                        If block1's x coordinate (dim 0) is greater than 0, use right arm to stack the block1, else use the left arm, and same for the block2.\
                        Note: You need to call the get_avoid_collision_pose function to avoid collisions when the left and right arms move alternately. \
                              For example, if the previous action uses the left arm and the next action uses the right arm, you need to move the left arm after release gripper to avoid collisions, vice versa.\
                              The pre-dis of stacked blocks may be smaller.",
    "current_code": """
                class gpt_stack_blocks_two(stack_blocks_two):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.block1": {
            "name": "block1",
            "description": "The first block to be stacked.",
            "modelname": "sapien-block1",
        },
        "self.block2": {
            "name": "block2",
            "description": "The second block to be stacked on top of the first block.",
            "modelname": "sapien-block1",
        },
        "self.block1_target_pose": {
            "name": "block1_target_pose",
            "description": "The target pose for the first block after stacking."
        }
    },
}

STACK_BLOCKS_THREE = {
    "task_name": "stack_blocks_three",
    "task_description":
    "Use the gripper to pick up block1 and move block 1 to the target position. Then pick up block 2 and place it on the block 1, and finally pick up\
                        block3 and place it on the block2.\
                        If block1's x coordinate (dim 0) is greater than 0, use right arm to stack the block1, else use the left arm. And same for the block2 and block3.\
                        Note: You need to call the get_avoid_collision_pose function to avoid collisions when the left and right arms move alternately. \
                              For example, if the previous action uses the left arm and the next action uses the right arm, you need to move the left arm after release gripper to avoid collisions, vice versa.\
                              The pre-dis of stacked blocks may be smaller.",
    "current_code": """
                class gpt_stack_blocks_three(stack_blocks_three):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.block1": {
            "name": "block1",
            "description": "The first block to be stacked.",
            "modelname": "sapien-block1",
        },
        "self.block2": {
            "name": "block2",
            "description": "The second block to be stacked on top of the first block.",
            "modelname": "sapien-block1",
        },
        "self.block3": {
            "name": "block3",
            "description": "The third block to be stacked on top of the second block.",
            "modelname": "sapien-block1",
        },
        "self.block1_target_pose": {
            "name": "block1_target_pose",
            "description": "The target pose for the first block after stacking."
        }
    },
}

PLACE_CONTAINER_PLATE = {
    "task_name": "place_container_plate",
    "task_description":
    "Use both arms to pick up the container and place it in the plate. If the container's x coordinate (dim 0) is greater than 0, \
                        use right arm to grasp the right side of the container, then pick up the container and place it in the plate. \
                        Else use the left arm grasp the left side of the container, then pick up the container and place it in the plate.\
                        Note: You may need to close the jaws tightly to pick up the container.",
    "current_code": """
                class gpt_place_container_plate(place_container_plate):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.container": {
            "name": "container",
            "description": "The container that needs to be placed in the plate.",
            "modelname": "002_bowl",
        },
        "self.plate": {
            "name": "plate",
            "description": "The plate where the container needs to be placed.",
            "modelname": "003_plate",
        }
    },
}

PLACE_EMPTY_CUP = {
    "task_name": "place_empty_cup",
    "task_description":
    "Use both arms to pick up the empty cup and place it on the coaster. If the cup's x coordinate (dim 0) is greater than 0, \
                        use right arm to grasp the cup, then pick up the cup and place it on the coaster,\
                        else use the left arm grasp the the cup, then pick up the cup and place it on the coaster.\
                        Note: You may need to close the jaws tightly to pick up the cup.\
                              Pre-dis for grabbing and placing cups may be smaller.\
                              The distance of lifting the cup may be smaller.",
    "current_code": """
                class gpt_place_empty_cup(place_empty_cup):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.cup": {
            "name": "cup",
            "description": "The empty cup that needs to be placed on the coaster.",
            "modelname": "021_cup",
        },
        "self.coaster": {
            "name": "coaster",
            "description": "The coaster where the empty cup needs to be placed.",
            "modelname": "019_coaster",
        }
    },
}

PLACE_SHOE = {
    "task_name": "place_shoe",
    "task_description":
    "Pick up the shoe and place it on the target block. And the head of the shoe should be towards the left side.\
                        The shoe is randomly placed on the table, if the shoe's x coordinate (dim 0) is greater than 0, use right arm to grasp the shoe, \
                        else use the left arm grasp the shoe.",
    "current_code": """
                class gpt_place_shoe(place_shoe):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.shoe": {
            "name": "shoe",
            "description": "The shoe that needs to be placed on the target block.",
            "modelname": "041_shoe",
        },
        "self.target_block": {
            "name": "target_block",
            "description": "The target block where the shoe needs to be placed.",
            "modelname": "sapien-block1",
        }
    },
}

################## Generalization Test Tasks ##################
# These tasks are used to evaluate the generalization ability of the code generation.
# No task-specific prompt tuning has been applied to them.

ADJUST_BOTTLE = {
    "task_name": "adjust_bottle",
    "task_description": "Pick up the bottle on the table headup with the correct arm.\
                        Move the arm upward by 0.1 meters along z-axis, and place the bottle at target pose.\
                        Note: You should keep gripper closed when placing the bottle.",
    "current_code": """
                class gpt_adjust_bottle(adjust_bottle):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.bottle": {
            "name": "bottle",
            "description": "The bottle should be picked up and placed at the target pose.",
            "modelname": "001_bottle"
        },
        "self.qpose_tag": {
            "name": "qpose_tag",
            "description": "A tag indicating which arm to use for picking up the bottle.\
                            0 means left arm, 1 means right arm.",
        },
        "self.left_target_pose": {
            "name": "left_target_pose",
            "description": "Target pose when use left arm to pick up the bottle.",
        },
        "self.right_target_pose": {
            "name": "right_target_pose",
            "description": "Target pose when use right arm to pick up the bottle.",
        }
    }
}

BLOCKS_RANKING_RGB= {
    "task_name": "blocks_ranking_rgb",
    "task_description": "Place the red block, green block, and blue block in the order of red, green, and blue from left to right, placing in a row.\
                        Pick and place each block to their target positions.\
                        Note: You should move end effector back to origin after placing each block to avoid collisions.\
                              You can place the red block, the green block, and the blue block in the order.",
    "current_code": """
                class gpt_blocks_ranking_rgb(blocks_ranking_rgb):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.block1": {
            "name": "box",
            "description": "Red block that should be placed on the left side.",
            "modelname": "sapien-block1",
        },
        "self.block2": {
            "name": "box",
            "description": "Green block that should be placed in the middle.",
            "modelname": "sapien-block1",
        },
        "self.block3": {
            "name": "box",
            "description": "Blue block that should be placed on the right side.",
            "modelname": "sapien-block1",
        },
        "self.block1_target_pose": {
            "name": "target_pose",
            "description": "Target pose for the red block.",
        },
        "self.block2_target_pose": {
            "name": "target_pose",
            "description": "Target pose for the green block.",
        },
        "self.block3_target_pose": {
            "name": "target_pose",
            "description": "Target pose for the blue block.",
        }
    }
}

BLOCKS_RANKING_SIZE = {
    "task_name": "blocks_ranking_size",
    "task_description": "There are three blocks on the table, the color of the blocks is random, move the blocks to the center of the table, and arrange them from largest to smallest, from left to right.\
                        Pick and place each block to their target positions.\
                        Note: You should move end effector back to origin after placing each block to avoid collisions.\
                              You can place the smallest block, the middle block, and the largest block in the order.",
    "current_code": """
                class gpt_blocks_ranking_size(blocks_ranking_size):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.block1": {
            "name": "box",
            "description": "The largest block that should be placed on the left side.",
            "modelname": "sapien-block1",
        },
        "self.block2": {
            "name": "box",
            "description": "The middle block that should be placed in the middle.",
            "modelname": "sapien-block1",
        },
        "self.block3": {
            "name": "box",
            "description": "The smallest block that should be placed on the right side.",
            "modelname": "sapien-block1",
        },
        "self.block1_target_pose": {
            "name": "target_pose",
            "description": "Target pose for the largest block.",
        },
        "self.block2_target_pose": {
            "name": "target_pose",
            "description": "Target pose for the middle block.",
        },
        "self.block3_target_pose": {
            "name": "target_pose",
            "description": "Target pose for the smallest block.",
        }
    }
}

CLICK_BELL = {
    "task_name": "click_bell",
    "task_description": "Click the bell's top center on the table.\
                        Move the top of bell's center and close gripper. And move the gripper down to touch the bell's top center.\
                        Note: You can change some API parameters to move above the bell's top center and close the gripper.\
                        You can use self.grasp_actor() to simulate the action of touch and click.\
                        self.grasp_actor() is only used to move the top center of the bell and close the gripper. So you must use same pre_grasp_dis and grasp_dis as the click_bell task.\
                        You don't need to lift the bell after clicking it, and you don't need to open the gripper or return the arm to origin position.",
    "current_code": """
                class gpt_click_bell(click_bell):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.bell": {
            "name": "bell",
            "description": "The bell that needs to be clicked.",
            "modelname": "050_bell",
        }
    }
}

MOVE_CAN_POT = {
    "task_name": "move_can_pot",
    "task_description": "There is a can and a pot on the table. Use one arm to pick up the can and move it to beside the pot.\
                        Grasp the can, and move the can upward. Place the can near the pot at target pose.\
                        Note: You don't need to return the arm to origin position. ",
    "current_code": """
                class gpt_move_can_pot(move_can_pot):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.can": {
            "name": "can",
            "description": "The can that needs to be moved to the pot.",
            "modelname": "105_sauce-can",
        },
        "self.pot": {
            "name": "pot",
            "description": "The pot at the center of the table.",
            "modelname": "060_kitchenpot",
        },
        "self.target_pose":{
            "name": "target_pose",
            "description": "The target pose where the can should be placed beside the pot.",
        }
    }
}

MOVE_PLAYINGCARD_AWAY = {
    "task_name": "move_playingcard_away",
    "task_description": "Use the arm to pick up the playing card and move it to left or right.\
                        Grasp the playing cards with specified arm, and then move the playing cards horizontally (right if right arm, left if left arm).\
                        Note: You should open gripper to release the playing cards after moving them.",
    "current_code": """
                class gpt_move_playingcard_away(move_playingcard_away):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.playingcards": {
            "name": "playingcards",
            "description": "The playing cards that need to be moved to left or right.",
            "modelname": "081_playingcards",
        }
    }
}

MOVE_STAPLER_PAD = {
    "task_name": "move_stapler_pad",
    "task_description": "Use appropriate arm to move the stapler to a colored mat.\
                        Grasp the stapler with specified arm, and move the arm upward. Place the stapler at target pose with alignment constraint.",
    "current_code": """
                class gpt_move_stapler_pad(move_stapler_pad):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.stapler": {
            "name": "stapler",
            "description": "The stapler that needs to be moved to the pad.",
            "modelname": "048_stapler",
        },
        "self.target_pose": {
            "name": "target",
            "description": "The target pose where the stapler should be placed on the pad."
        }
    }
}

CLICK_ALARMCLOCK = {
    "task_name": "click_alarmclock",
    "task_description": "Click the alarm clock's center of the top side button on the table.\
                        Move the top of bell's center and close gripper. And move the gripper down.\
                        Note: You can change some API parameters to move above the alarm clock's top center and close the gripper(grasp_actor).\
                        You can use self.grasp_actor() to simulate the action of touch and click",
    "current_code": """
                class gpt_click_alarmclock(click_alarmclock):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.alarm": {
            "name": "alarm",
            "description": "The alarm clock that needs to be clicked.",
            "modelname": "046_alarm-clock",
        }
    }
}

MOVE_PILLBOTTLE_PAD = {
    "task_name": "move_pillbottle_pad",
    "task_description": "Use one arm to pick the pillbottle and place it onto the pad.\
                        Grasp the pillbottle. Get the target pose for placing the pillbottle, and place the pillbottle at the target pose.",
    "current_code": """
                class gpt_move_pillbottle_pad(move_pillbottle_pad):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.pillbottle": {
            "name": "pillbottle",
            "description": "The pillbottle that needs to be moved to the pad.",
            "modelname": "080_pillbottle",
        },
        "self.pad": {
            "name": "pad",
            "description": "The pad where the pillbottle should be placed.",
            "modelname": "sapien-block1",
        },
    }
}

PLACE_A2B_LEFT = {
    "task_name": "place_a2b_left",
    "task_description": "Use appropriate arm to place object on the left of target object.\
                        Grasp the object with specified arm. And get target pose and adjust x position to place object to the left of target object.\
                        Place the object at the adjusted target object position.\
                        Note: You can decrease the x position of target pose by 0.13 to place object to the left of target object. (target_pose[0] -= 0.13)",
    "current_code": """
                class gpt_place_a2b_left(place_a2b_left):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.object": {
            "name": "object",
            "description": "The object that needs to be placed on the left of the target object.",
            "modelname": None,  # Replace with actual model name
        },
        "self.target_object": {
            "name": "target_object",
            "description": "The target object where the object should be placed to its left, you can get the target pose from this object by target_pose = self.target_object.get_pose().p.tolist()",
            "modelname": None,  # Replace with actual model name
        },
    }
}

PLACE_A2B_RIGHT = {
    "task_name": "place_a2b_right",
    "task_description": "Use appropriate arm to place object on the right of target object.\
                        Grasp the object with specified arm. And get target pose and adjust x position to place object to the right of target object.\
                        Place the object at the adjusted target object position.\
                        Note: You can increase the x position of target pose by 0.13 to place object to the right of target object. (target_pose[0] += 0.13)",
    "current_code": """
                class gpt_place_a2b_right(place_a2b_right):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.object": {
            "name": "object",
            "description": "The object that needs to be placed on the right of the target object.",
            "modelname": None,  # Replace with actual model name
        },
        "self.target_object": {
            "name": "target_object",
            "description": "The target object where the object should be placed to its left, you can get the target pose from this object by target_pose = self.target_object.get_pose().p.tolist()",
            "modelname": None,  # Replace with actual model name
        },
    }
}

PLACE_BREAD_BASKET = {
    "task_name": "place_bread_basket",
    "task_description": "If there is one bread on the table, use one arm to grab the bread and put it in the basket. If there are two breads on the table, use two arms to simultaneously grab up two breads and put them in the basket.\
                        Grasp the bread. If there is one bread, place the bread into the basket. If there is two breads, place left bread into the basket, and place right bread into the basket when move left arm back to origin.\
                        Note: You should move the arm back to origin after placing the bread to avoid collisions.",
    "current_code": """
                class gpt_place_bread_basket(place_bread_basket):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.bread[id]": {
            "name": "bread[id]",
            "description": "A list of breads that need to be placed in the basket. If there is one bread, id=0. If there are two breads, id=0 and id=1.",
            "modelname": "075_bread",
        },
        "self.breadbasket": {
            "name": "breadbasket",
            "description": "The basket where the bread needs to be placed.",
            "modelname": "076_breadbasket",
        },
    }
}

PLACE_CAN_BASKET = {
    "task_name": "place_can_basket",
    "task_description": "Use one arm to pick up the can and place it into the basket. Use the other arm to lift up the basket.\
                        Grasp the can with the specified arm. Place the can at the selected position into the basket. Lift the basket with the opposite arm.\
                        Note: You should not open the gripper after lifting the basket.\
                              The height of lifting the basket is 5 cm.",
    "current_code": """
                class gpt_place_can_basket(place_can_basket):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.can": {
            "name": "can",
            "description": "The can that needs to be placed in the basket.",
            "modelname": "071_can",
        },
        "self.basket": {
            "name": "basket",
            "description": "The basket where the can needs to be placed.",
            "modelname": "110_basket",
        },
        "self.get_arm_pose(arm_tag=self.arm_tag)": {
            "name": "place_pose",
            "description": "The target pose where the can should be placed in the basket.",
            "modelname": None,
        }
    }
}

PLACE_FAN = {
    "task_name": "place_fan",
    "task_description": "Grab the fan and place it on a colored pad.\
                        Grasp the fan with the selected arm. Place the fan to the target pose.\
                        Note: The height of lifting the fan is small. Fan have front and back, so you should use constraint 'align' to align the fan's front with the pad's front.",
    "current_code": """
                class gpt_place_fan(place_fan):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.fan": {
            "name": "fan",
            "description": "The fan that needs to be placed on the pad.",
            "modelname": "099_fan",
        },
        "self.target_pose": {
            "name": "target_pose",
            "description": "The target pose where the fan should be placed on the pad.",
            "modelname": None,
        }
    }
}

PLACE_MOUSE_PAD = {
    "task_name": "place_mouse_pad",
    "task_description": "Grasp the mouse and place it on a colored pad.\
                        Grasp the mouse with the selected arm. Place the mouse at the target location.\
                        Note: The mouse have front and back, so you should use constraint 'align' to align the mouse's front with the pad's front.",
    "current_code": """
                class gpt_place_mouse_pad(place_mouse_pad):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.mouse": {
            "name": "mouse",
            "description": "The mouse that needs to be placed on the pad.",
            "modelname": "047_mouse",
        },
        "self.target_pose": {
            "name": "target_pose",
            "description": "The target pose where the mouse should be placed on the pad.",
            "modelname": None,
        }
    }
}

PLACE_OBJECT_BASKET = {
    "task_name": "place_object_basket",
    "task_description": "Use one arm to grab the target object and put it in the basket, then use the other arm to grab the basket, and finally move the basket slightly away.\
                        Grasp the object with the specified arm. Place the object at the selected position into the basket. Lift the basket with the opposite arm.\
                        Note: You should not open the gripper after lifting the basket.\
                              The height of lifting the basket is 5 cm.",
    "current_code": """
                class gpt_place_object_basket(place_object_basket):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.object": {
            "name": "object",
            "description": "The object that needs to be placed in the basket.",
            "modelname": None,  # Replace with actual model name
        },
        "self.basket": {
            "name": "basket",
            "description": "The basket where the object needs to be placed.",
            "modelname": "110_basket",
        },
    }
}

PLACE_OBJECT_SCALE = {
    "task_name": "place_object_scale",
    "task_description": "Use one arm to grab the object and put it on the scale.\
                        Grasp the object with the selected arm. Place the object on the scale.\
                        Note: Don't use functional_point_id and pre_dis_axis='fp', because the object can be any object that is specified in the task.",
    "current_code": """
                class gpt_place_object_scale(place_object_scale):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.object": {
            "name": "object",
            "description": "The object that needs to be placed on the scale.",
            "modelname": None, # The object can be any object that is specified in the task  
        },
        "self.scale": {
            "name": "scale",
            "description": "The scale where the object needs to be placed.",
            "modelname": "072_electronicscale",
        },
    }
}

PLACE_OBJECT_STAND = {
    "task_name": "place_object_stand",
    "task_description": "Use appropriate arm to place the object on the stand.\
                        Grasp the object with the specified arm. Place the object onto the display stand.\
                        Note: Don't use functional_point_id and pre_dis_axis='fp', because the object can be any object that is specified in the task.",
    "current_code": """
                class gpt_place_object_stand(place_object_stand):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.object": {
            "name": "object",
            "description": "The object that needs to be placed on the stand.",
            "modelname": None,  # The object can be any object that is specified in the task
        },
        "self.displaystand": {
            "name": "displaystand",
            "description": "The display stand where the object needs to be placed.",
            "modelname": "074_displaystand",
        }
    }
}

PLACE_PHONE_STAND = {
    "task_name": "place_phone_stand",
    "task_description": "Pick up the phone and put it on the phone stand.\
                        Grasp the phone with specified arm. Place the phone onto the stand's functional point and align the points.",
    "current_code": """
                class gpt_place_phone_stand(place_phone_stand):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.phone": {
            "name": "phone",
            "description": "The phone that needs to be placed on the stand.",
            "modelname": "077_phone",
        },
        "self.stand": {
            "name": "stand",
            "description": "The phone stand where the phone needs to be placed.",
            "modelname": "078_phonestand",
        },
    }
}

PRESS_STAPLER = {
    "task_name": "press_stapler",
    "task_description": "Use one arm to press the stapler.\
                        Move arm to the position of the stapler and close the gripper. Move the stapler down slightly.\
                        Note: You can use self.grasp_actor() to simulate the action of move to the position of stapler or pressing the stapler.\
                        The stapler should be pressed at the top center.",
    "current_code": """
                class gpt_press_stapler(press_stapler):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.stapler": {
            "name": "stapler",
            "description": "The stapler that needs to be pressed.",
            "modelname": "048_stapler",
        }
    }
}

ROTATE_QRCODE = {
    "task_name": "rotate_qrcode",
    "task_description": "Use arm to catch the qrcode board on the table, pick it up and rotate to let the qrcode face towards you.\
                        Grasp the QR code with specified pre-grasp distance. Place the QR code at the target position.\
                        Note: The QR code have front and back, so you should use constraint 'align' to align the QR code's front with the target position.\
                        Don't use functional point of the QR code when placing it.",
    "current_code": """
                class gpt_rotate_qrcode(rotate_qrcode):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.qrcode": {
            "name": "qrcode",
            "description": "The QR code sign that needs to be rotated.",
            "modelname": "070_paymentsign",
        },
        "self.target_pose": {
            "name": "target_pose",
            "description": "The target pose where the QR code should be placed.",
            "modelname": None,  # No specific model for this pose
        }
    }
}

STACK_BOWLS_THREE = {
    "task_name": "stack_bowls_three",
    "task_description": "Stack the three bowls on top of each other.\
                        Move bowl 1 to the target pose, then move bowl 2 above bowl 1, and finally move bowl 3 above bowl 2.\
                        Note: The target pose of bowl 2 is at 5 cm above bowl 1, and the target pose of bowl 3 is at 5 cm above bowl 2.\
                              All target pose is np.ndarray([x, y, z]), so you should concatenate the quaternion later.",
    "current_code": """
                class gpt_stack_bowls_three(stack_bowls_three):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.bowl1": {
            "name": "bowl1",
            "description": "The first bowl that should be placed at the bottom, you can get bowl1's position by using self.bowl1.get_pose().p.",
            "modelname": "002_bowl",
        },
        "self.bowl2": {
            "name": "bowl2",
            "description": "The second bowl that should be placed above the first bowl, you can get bowl1's position by using self.bowl1.get_pose().p, you can get the target pose of bowl 2 by adding 5 cm to the z-axis of bowl 1's target pose.",
            "modelname": "002_bowl",
        },
        "self.bowl3": {
            "name": "bowl3",
            "description": "The third bowl that should be placed above the second bowl, you can get bowl2's position by using self.bowl2.get_pose().p, you can get the target pose of bowl 3 by adding 5 cm to the z-axis of bowl 2's target pose.",
            "modelname": "002_bowl",
        },
        "self.bowl1_target_pose": {
            "name": "bowl1_target_pose",
            "description": "The target pose for the first bowl. It's a numpy.ndarray([x, y, z]) that should use .tolist() to be concatenated with the quaternion later.",
            "modelname": None,  # No specific model for this pose
        },
        "self.quat_of_target_pose": {
            "name": "quat_of_target_pose",
            "description": "The quaternion of the target pose for the bowls, To be concatenated with the target pose.",
            "modelname": None,  # No specific model for this pose
        },
    }
}

STACK_BOWLS_TWO = {
    "task_name": "stack_bowls_two",
    "task_description": "Stack the two bowls on top of each other.\
                        Move bowl 1 to the target pose, then move bowl 2 above bowl 1.\
                        Note: The target pose of bowl 2 is at 5 cm above bowl 1.\
                              All target pose is np.ndarray([x, y, z]), so you should concatenate the quaternion later.",
    "current_code": """
                class gpt_stack_bowls_two(stack_bowls_two):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.bowl1": {
            "name": "bowl1",
            "description": "The first bowl that should be placed at the bottom, you can get bowl1's position by using self.bowl1.get_pose().p.",
            "modelname": "002_bowl",
        },
        "self.bowl2": {
            "name": "bowl2",
            "description": "The second bowl that should be placed above the first bowl, you can get bowl1's position by using self.bowl1.get_pose().p, you can get the target pose of bowl 2 by adding 5 cm to the z-axis of bowl 1's target pose.",
            "modelname": "002_bowl",
        },
        "self.bowl1_target_pose": {
            "name": "bowl1_target_pose",
            "description": "The target pose for the first bowl. It's a numpy.ndarray([x, y, z]) that should use .tolist() to be concatenated with the quaternion later.",
            "modelname": None,  # No specific model for this pose
        },
        "self.quat_of_target_pose": {
            "name": "quat_of_target_pose",
            "description": "The quaternion of the target pose for the bowls, To be concatenated with the target pose.",
            "modelname": None,  # No specific model for this pose
        },
    }
}

#Note: You would better grasp the seal from top down direction.

STAMP_SEAL = {
    "task_name": "stamp_seal",
    "task_description": "Use one arm to pick the stamp and place it on the target block.\
                        Grasp the seal with specified arm. Place the seal on the target block.\
                        Note: Don't set pre_dis_axis to fp, because the pre_dis_axis is not used in this task.",
    "current_code": """
                class gpt_stamp_seal(stamp_seal):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.seal": {
            "name": "seal",
            "description": "The seal that needs to be placed on the target block.",
            "modelname": "100_seal",
        },
        "self.target_pose": {
            "name": "target_pose",
            "description": "The target pose where the seal should be placed on the target block.",
            "modelname": None,  # No specific model for this pose
        }
    }
}

SHAKE_BOTTLE_HORIZONTALLY = {
    "task_name": "shake_bottle_horizontally",
    "task_description": "Shake the bottle horizontally with proper arm.\
                        Grasp the bottle with specified arm. Shake the bottle horizontally by moving the arm left and right.",
    "current_code": """
                class gpt_shake_bottle_horizontally(shake_bottle_horizontally):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.bottle": {
            "name": "bottle",
            "description": "The bottle that needs to be shaken horizontally.",
            "modelname": "001_bottle", 
        }
    }
}

SHAKE_BOTTLE = {
    "task_name": "shake_bottle",
    "task_description": "Shake the bottle with proper arm.\
                        Grasp the bottle with specified arm. Shake the bottle by moving the arm up and down.",
    "current_code": """
                class gpt_shake_bottle(shake_bottle):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.bottle": {
            "name": "bottle",
            "description": "The bottle that needs to be shaken.",
            "modelname": "001_bottle",
        }
    }
}

TURN_SWITCH = {
    "task_name": "turn_switch",
    "task_description": "Use one arm to click the switch.\
                        Close the gripper before clicking the switch. Then move the arm to the switch and click it.\
                        Note: You can use grasp_actor() to simulate the action of clicking the switch.",
    "current_code": """
                class gpt_turn_switch(turn_switch):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.switch": {
            "name": "switch",
            "description": "The switch that needs to be turned on or off.",
            "modelname": "056_switch",
        }
    }
}

OPEN_LAPTOP = {
    "task_name": "open_laptop",
    "task_description": "Open the laptop with one proper arm.\
                        Grasp the laptop with specified arm. Open the laptop by moving the arm up.",
    "current_code": """
                class gpt_open_laptop(open_laptop):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.laptop": {
            "name": "laptop",
            "description": "The laptop that needs to be opened.",
            "modelname": "015_laptop",
        }
    }
}

OPEN_MICROWAVE = {
    "task_name": "open_microwave",
    "task_description": "Open the microwave with one proper arm.\
                        Grasp the handle of the microwave with specified arm. Pull the handle to open the microwave",
    "current_code": """
                class gpt_open_microwave(open_microwave):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.microwave": {
            "name": "microwave",
            "description": "The microwave that needs to be opened.",
            "modelname": "044_microwave",
        }
    }
}

TURN_SWITCH = {
    "task_name": "turn_switch",
    "task_description": "Use one arm to click the switch.\
                        Close the gripper before clicking the switch. Then move the arm to the switch and click it.\
                        Note: You can use grasp_actor() to simulate the action of clicking the switch.",
    "current_code": """
                class gpt_turn_switch(turn_switch):
                    def play_once(self):
                        pass
                """,
    "actor_list": {
        "self.switch": {
            "name": "switch",
            "description": "The switch that needs to be turned on or off.",
            "modelname": "056_switch",
        }
    }
}

def get_all_tasks():
    return {
        key: value
        for key, value in globals().items()
        if key.isupper() and isinstance(value, dict) and value  # value非空dict
    }

