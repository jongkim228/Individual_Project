from init import *

def bounding_box(box_size, rotation):

    x,y,z = box_size

    if rotation == "default":
        y_width = y + LEFT_FINGER_THICKNESS + RIGHT_FINGER_THICKNESS
        bounding_box = np.array([x, y_width, z])
        return bounding_box

    elif rotation == "z_rotation":
        x_width = x + LEFT_FINGER_THICKNESS + RIGHT_FINGER_THICKNESS
        bounding_box = np.array([x_width, y, z])
        return bounding_box
    
def territory_calculation(placed_boxes,solution):
    placed_boxes_territory = []
    for i, pack_pos in zip(placed_boxes, solution):
        geom_id = model.body_geomadr[i]
        box_size = model.geom_size[geom_id]

        solution = np.array([pack_pos["x"], pack_pos["y"], pack_pos["z"]])

        max = solution + box_size
        min = solution - box_size

        placed_boxes_territory.append({
            "id": i,
            "min": min,
            "max": max
        })
        
    return placed_boxes_territory

        
def collision_check(target_box, rotation, placed_boxes,solution):
    #placed boxes territory (np.array)
    placed_boxes_territory = territory_calculation(placed_boxes)
    #target box
    geom_id = model.body_geomadr[target_box]
    box_size = model.geom_size[geom_id]

    total_box = bounding_box(box_size,rotation)
    checking_box = total_box + solution

    for box in placed_boxes_territory:
        x_min = box["min"][0]
        x_max = box["max"][0]

        y_min = box["min"][1]
        y_max = box["max"][1]

        z_min = box["min"][2]
        z_max = box["max"][2]
        
        x_overlap = checking_box[0] < x_max and x_min < checking_box[0]
        y_overlap = checking_box[1] < y_max and y_min < checking_box[1]
        z_overlap = checking_box[2] < z_max and z_min < checking_box[2]

        if x_overlap and y_overlap and z_overlap:
            return "collison"
        else:
            return "safe"

        



