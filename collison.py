from init import *

def bounding_box(box_size, rotation):
    print(rotation)
    x, y, z = box_size

    if rotation == "z_90_rotated":
        x_width = x + LEFT_FINGER_THICKNESS + RIGHT_FINGER_THICKNESS
        return np.array([x_width, y, z])
    else:
        y_width = y + LEFT_FINGER_THICKNESS + RIGHT_FINGER_THICKNESS
        return np.array([x,y_width, z])

# placed boxes coordinates
def territory_calculation(placed_boxes, solutions):
    placed_boxes_territory.clear()
    offset = np.zeros(3)

    for box_id, solution in zip(placed_boxes,solutions):
        geom_id = model.body_geomadr[box_id]
        box_size = model.geom_size[geom_id]
        box_pos = data.geom_xpos[geom_id]

        solver_pos = np.array([solution["x"], solution["y"], solution["z"]])
        offset = box_pos - solver_pos

        max_pos = box_pos + box_size
        min_pos = box_pos - box_size

        placed_boxes_territory.append({
            "id": box_id,
            "min": min_pos,
            "max": max_pos
        })

    return placed_boxes_territory, offset


def collision_check(target_box, rotation, placed_boxes, box_solution,solutions):

    # calculation for placed boxes
    territory, offset = territory_calculation(placed_boxes,solutions)

    # target box
    geom_id = model.body_geomadr[target_box]
    box_size = model.geom_size[geom_id]

    # bound box with gripper
    bounded_box = bounding_box(box_size, rotation)
    print(f"bounded_box: {bounded_box}")

    # if box is placed on target place
    if len(territory) > 0:
        solution_center = np.array([box_solution["x"], box_solution["y"], box_solution["z"]]) + offset

        # calculation for target box with solution
        bound_max = solution_center + bounded_box
        bound_min = solution_center - bounded_box

        print(f"bounded_box: {bounded_box}")
        print(f"bound_min: {bound_min}")
        print(f"bound_max: {bound_max}")

        for box in territory:
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

                if rotation == "z_90_rotated":
                    other_rotation = "default"
                else:
                    other_rotation = "z_90_rotated"
                
                new_bounded = bounding_box(box_size, other_rotation)
                other_max = solution_center + new_bounded
                other_min = solution_center - new_bounded
                
                ox = other_min[0] < x_max and other_max[0] > x_min
                oy = other_min[1] < y_max and other_max[1] > y_min
                oz = other_min[2] < z_max and other_max[2] > z_min


                
                
                
                

    return "safe"