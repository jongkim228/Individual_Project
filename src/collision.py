from init import *

def bounding_box(box_size, grip_dir):

    x, y, z = box_size
    finger_thickness = LEFT_FINGER_THICKNESS + RIGHT_FINGER_THICKNESS

    if grip_dir == "x_axis":
        x_half = x + finger_thickness
        return np.array([x_half, y, z])
    else:
        y_half = y + finger_thickness
        return np.array([x, y_half, z])


def territory_calculation(placed_boxes, solutions):
    placed_boxes_territory.clear()
    offsets = []

    for box_id, solution in zip(placed_boxes, solutions):
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


def collision_check(target_box, grip_dir, placed_boxes, box_solution, solutions):
    collide = False
    territory, offsets = territory_calculation(placed_boxes, solutions)

    geom_id = model.body_geomadr[target_box]
    box_size = model.geom_size[geom_id].copy()

    bounded_box = bounding_box(box_size, grip_dir)


    if len(territory) == 0:
        return "safe", grip_dir

    solution_center = np.array([box_solution["x"], box_solution["y"], box_solution["z"]])
    target_center = solution_center
    target_box_bottom = target_center[2] - box_size[2]

    same_floor_territory = []
    for box in territory:
        placed_box_bottom = box["min"][2]
        height_diff = abs(placed_box_bottom - target_box_bottom)



        if height_diff < box_size[2]:
            same_floor_territory.append(box)

    bound_max = target_center + bounded_box
    bound_min = target_center - bounded_box

    for box in same_floor_territory:
        x_overlap = bound_min[0] < box["max"][0] and bound_max[0] > box["min"][0]
        y_overlap = bound_min[1] < box["max"][1] and bound_max[1] > box["min"][1]
        z_overlap = bound_min[2] < box["max"][2] and bound_max[2] > box["min"][2]


        if x_overlap and y_overlap and z_overlap:
            collide = True
            break

    if collide:
        other_dir = "y_axis" if grip_dir == "x_axis" else "x_axis"
        new_bounded = bounding_box(box_size, other_dir)
        bound_max = target_center + new_bounded
        bound_min = target_center - new_bounded

        other_collide = False
        for box in same_floor_territory:
            x_overlap = bound_min[0] < box["max"][0] and bound_max[0] > box["min"][0]
            y_overlap = bound_min[1] < box["max"][1] and bound_max[1] > box["min"][1]
            z_overlap = bound_min[2] < box["max"][2] and bound_max[2] > box["min"][2]
            if x_overlap and y_overlap and z_overlap:
                other_collide = True
                break

        if other_collide:
            return "drop", grip_dir
        else:
            return "rotate", other_dir

    return "safe", grip_dir


def finger_contact(data, finger_collision_geom_ids):
    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if g1 in finger_collision_geom_ids or g2 in finger_collision_geom_ids:
            return True
    return False

def box_contact(data, model, placed_boxes, target_box_id):
    placed_geom_ids = set()
    for p in placed_boxes:
        for i in range(model.ngeom):
            if model.geom(i).bodyid == p:
                placed_geom_ids.add(i)

    target_geom_ids = set()
    for i in range(model.ngeom):
        if model.geom(i).bodyid == target_box_id:
            target_geom_ids.add(i)

    for i in range(data.ncon):
        g1, g2 = data.contact[i].geom1, data.contact[i].geom2
        if g1 in target_geom_ids and g2 in placed_geom_ids:
            return True
        if g2 in target_geom_ids and g1 in placed_geom_ids:
            return True
        
    return False
