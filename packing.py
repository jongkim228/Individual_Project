import mujoco
import csv
import subprocess


SCALE = 1000
MARGIN = 0.007
height = 0.07
max_height = 0.5

def calculate_area_usage(file_name,floor_area):
    total_volume = 0
    with open(file_name,"r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["TYPE"] == "ITEM":
                lx = int(row["LX"]) / SCALE
                ly = int(row["LY"]) / SCALE
                lz = int(row["LZ"]) / SCALE
                total_volume += lx*ly *lz
        
    return total_volume


def box_solution(data, model, boxes, placed_boxes):
    target_space_id = model.body("target_space").id
    target_pos = data.xpos[target_space_id]

    geom_id = model.body_geomadr[target_space_id]
    size = model.geom_size[geom_id]
    length = size[0] * 2
    width = size[1] * 2

    floor_area = length * width

    total_area = 0
    for i in placed_boxes:
        box_geom_id = model.body_geomadr[i]
        located_box_size = model.geom_size[box_geom_id]
        total_area += located_box_size[0] * located_box_size[1] * 4

    for i in boxes:
        box_geom_id = model.body_geomadr[i]
        located_box_size = model.geom_size[box_geom_id]
        total_area += located_box_size[0] * located_box_size[1] * 4

    area_usage = total_area / floor_area

    csv_box = []
    for body_id in boxes:
        geom_id = model.body_geomadr[body_id]
        box_size = model.geom_size[geom_id] * 2
        csv_box.append(box_size)

    with open("items.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["X", "Y", "Z", "ROTATIONS", "COPIES"])
        writer.writeheader()
        for x, y, z in csv_box:

            if x <= length and y <= width:
                rotations = 1
            else:
                rotations = 63

            writer.writerow({
                "X": int((x + MARGIN * 2) * SCALE),
                "Y": int((y + MARGIN * 2) * SCALE),
                "Z": int(z * SCALE),
                "ROTATIONS": rotations,
                "COPIES": 1
            })


    layer_height = max(max(box) for box in csv_box)
    height_local = layer_height

    origin_x = target_pos[0] + length / 2 
    origin_y = target_pos[1] + width / 2 
    origin_z = target_pos[2] + 0.001


    while True:

        with open("bins.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["X", "Y", "Z"])
            writer.writeheader()
            writer.writerow({
                "X": int(length * SCALE),
                "Y": int(width * SCALE),
                "Z": int(height_local * SCALE)
            })

        with open("parameters.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["NAME", "VALUE"])
            writer.writeheader()
            writer.writerow({
                "NAME": "objective", "VALUE": "bin-packing"
            })

        result = subprocess.run([
            "./packingsolver/build/src/box/packingsolver_box",
            "--items", "items.csv",
            "--verbosity-level", "1",
            "--bins", "bins.csv",
            "--parameters", "parameters.csv",
            "--certificate", "solutions.csv",
            "--time-limit", "10",
            "--objective", "BinPacking",
            "--optimization-mode", "NotAnytimeSequential",
            "--use-sequential-single-knapsack", "true",
            "--use-tree-search", "false",
            "--use-sequential-value-correction", "false",
            "--use-column-generation", "false",
            "--use-dichotomic-search", "false"
        ], capture_output=True, text=True)

        area_usage = calculate_area_usage("solutions.csv", floor_area)
        if area_usage >= 0.8 or height_local >= max_height:
            break

        height_local += max(box[2] for box in csv_box)

    results = []
    with open("solutions.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["TYPE"] == "ITEM":
                item_id = int(row["ID"])
                box_size = csv_box[item_id]
                
                solver_x = int(row["X"]) / SCALE
                solver_y = int(row["Y"]) / SCALE
                solver_z = int(row["Z"]) / SCALE

                solver_lx = int(row["LX"]) / SCALE - MARGIN * 2
                solver_ly = int(row["LY"]) / SCALE - MARGIN * 2
                solver_lz = int(row["LZ"]) / SCALE - MARGIN * 2

                x_local = solver_x + solver_lx / 2  + MARGIN
                y_local = solver_y + solver_ly / 2  + MARGIN
                z_local = solver_z + solver_lz / 2

                world_x = origin_x - x_local
                world_y = origin_y - y_local
                world_z = origin_z + z_local

                rotation = int(row.get("ROTATION", 0))


                results.append({
                    "id":       item_id,
                    "x":        world_x,
                    "y":        world_y,
                    "z":        world_z,
                    "rotation": rotation
                })

    results.sort(key=lambda r: r["z"])
    bin_volume = length * width * height_local
    total_box_volume = calculate_area_usage("solutions.csv", floor_area)
    packing_efficiency = total_box_volume / bin_volume
    print(f"Packing efficiency: {packing_efficiency*100:.1f}%")
    return results