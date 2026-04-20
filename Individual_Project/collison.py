from init import *

def bounding_box(box_size, rotation):
    print(rotation)
    x,y,z = box_size

    if rotation == "long":
        x_width = x + LEFT_FINGER_THICKNESS + RIGHT_FINGER_THICKNESS
        bounding_box = np.array([x_width,y, z])
        return bounding_box
    else:
        y_width = y + LEFT_FINGER_THICKNESS + RIGHT_FINGER_THICKNESS
        bounding_box = np.array([x, y_width, z])
        return bounding_box
    
#placed boxes coordinates
def territory_calculation(placed_boxes):
    placed_boxes_territory.clear()
    for i in zip(placed_boxes):
        geom_id = model.body_geomadr[i]
        box_size = model.geom_size[geom_id]
        box_pos = data.geom_xpos[geom_id]

        max = box_pos + box_size
        min = box_pos - box_size

        placed_boxes_territory.append({
            "id": i,
            "min": min,
            "max": max
        })

    return placed_boxes_territory

        
def collision_check(target_box, rotation, placed_boxes,box_solution,solutions):

    #calculation for placed boxes
    territory = territory_calculation(placed_boxes)

    #target box
    geom_id = model.body_geomadr[target_box]
    box_size = model.geom_size[geom_id]
    #bound box with gripper
    bounded_box = bounding_box(box_size,rotation)

    #if box is placed on target place
    if len(territory) > 0:
        solution_center = np.array([box_solution["x"],box_solution["y"],box_solution["z"]])
        
        #calculation for target box with solution
        bound_max = solution_center + bounded_box
        bound_min = solution_center - bounded_box

        for box in territory:
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
                return "collison"

    
    return "safe"     



