from init import *

def bounding_box(box_size, rotation):

    x,y,z = box_size

    if rotation == "default":
        y_width = y + LEFT_FINGER_THICKNESS + RIGHT_FINGER_THICKNESS
        bounding_box = x + y_width + z
        return bounding_box

    elif rotation == "z_rotation":
        x_width = x + LEFT_FINGER_THICKNESS + RIGHT_FINGER_THICKNESS
        bounding_box = x_width + y + z
        return bounding_box
    
def territory_calculation(sorted_boxes):
    for i in sorted_boxes:
        geom_id = model.body_geomadr[i]
        box_size = model.geom_size[geom_id]
        box_pos = data.xpos[i]

        territory = [a + b for a,b in zip(box_size,box_pos)]
        territory_list = []
        territory_list.append(territory)

    return territory_list

        
def collision_check(model,box_id,solution,rotation):
    
    geom_id = model.body_geomadr[box_id]
    box_size = model.geom_size[geom_id]

    bounding_box(box_size,rotation)


