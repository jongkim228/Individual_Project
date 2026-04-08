from init import *

def bounding_box(box_size, grip_dir):

    x, y, z = box_size

    if grip_dir == "x_axis":
        x_width = x + LEFT_FINGER_THICKNESS + RIGHT_FINGER_THICKNESS
        return np.array([x_width, y, z])
    else:
        y_width = y + LEFT_FINGER_THICKNESS + RIGHT_FINGER_THICKNESS
        return np.array([x,y_width, z])

# placed boxes coordinates
def territory_calculation(placed_boxes, solutions):
    placed_boxes_territory.clear()
    offsets = []

    for box_id, solution in zip(placed_boxes,solutions):
        geom_id = model.body_geomadr[box_id]
        box_size = model.geom_size[geom_id]
        box_pos = data.geom_xpos[geom_id]

        solver_pos = np.array([solution["x"], solution["y"], solution["z"]])
        offset = box_pos - solver_pos
        offsets.append(offset)

        max_pos = box_pos + box_size
        min_pos = box_pos - box_size

        placed_boxes_territory.append({
            "id": box_id,
            "min": min_pos,
            "max": max_pos
        })

    return placed_boxes_territory, offsets


def collision_check(target_box, grip_dir, placed_boxes, box_solution,solutions):
    collide = False
    # calculation for placed boxes
    territory, offsets = territory_calculation(placed_boxes,solutions)

    # target box
    geom_id = model.body_geomadr[target_box]
    box_size = model.geom_size[geom_id].copy()

    # bound box with gripper
    bounded_box = bounding_box(box_size, grip_dir)

    print("territory len",len(territory))

    if len(territory) == 0:
        return "safe", grip_dir

    # if box is placed on target place

    if len(territory) > 0:

        avg_offset = np.mean(offsets, axis = 0)
        solution_center = np.array([box_solution["x"], box_solution["y"], box_solution["z"]])
        target_center = solution_center + avg_offset

        same_floor_territory = []
        for box in territory:
            placed_box_bottom = box["min"][2]
            target_box_bottom = target_center[2] - box_size[2]
            height_diff = abs(placed_box_bottom - target_box_bottom)

            if height_diff < box_size[2]:
                same_floor_territory.append(box)


        # calculation for target box with solution
        bound_max = target_center + bounded_box
        bound_min = target_center - bounded_box

        print(f"bounded_box: {bounded_box}")
        print(f"bound_min: {bound_min}")
        print(f"bound_max: {bound_max}")

        for box in same_floor_territory:
            print(f"placed box min: {box['min']}, max: {box['max']}")

            x_min = box["min"][0]
            x_max = box["max"][0]

            y_min = box["min"][1]
            y_max = box["max"][1]

            z_min = box["min"][2]
            z_max = box["max"][2]

            x_overlap = bound_min[0] < x_max and bound_max[0] > x_min
            y_overlap = bound_min[1] < y_max and bound_max[1] > y_min
            z_overlap = bound_min[2] < z_max and bound_max[2] > z_min

            print(f"x_overlap: {x_overlap}, y_overlap: {y_overlap}, z_overlap: {z_overlap}")

            if x_overlap and y_overlap and z_overlap:
                collide = True
                break

        if collide:

            if grip_dir == "x_axis":  
                other_dir = "y_axis"
            else:
                other_dir = "x_axis"   

            new_bounded = bounding_box(box_size, other_dir)
            bound_max = target_center + new_bounded
            bound_min = target_center - new_bounded

            other_collide = False

            for box in same_floor_territory:
                x_min = box["min"][0]
                x_max = box["max"][0]

                y_min = box["min"][1]
                y_max = box["max"][1]

                z_min = box["min"][2]
                z_max = box["max"][2]

                x_overlap = bound_min[0] < x_max and bound_max[0] > x_min
                y_overlap = bound_min[1] < y_max and bound_max[1] > y_min
                z_overlap = bound_min[2] < z_max and bound_max[2] > z_min

                if x_overlap and y_overlap and z_overlap:
                    other_collide = True
                    break
            if other_collide:
                return "drop", grip_dir
            else:
                return "rotate", other_dir
            
        
    return "safe", grip_dir
