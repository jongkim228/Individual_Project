import mujoco
import csv
import subprocess

SCALE = 1000
MARGIN = 0.005
MAX_LEVELS = 5 


def calculate_area_usage(file_name, floor_area):
    total_area = 0
    with open(file_name, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["TYPE"] == "ITEM":
                lx = int(row["LX"]) / SCALE
                ly = int(row["LY"]) / SCALE
                total_area += lx * ly
    return total_area / floor_area


def run_packing_solver(remaining_boxes, length, width, layer_height):


    with open("items.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["X", "Y", "Z", "ROTATIONS", "COPIES"])
        writer.writeheader()
        for x, y, z in remaining_boxes:
            writer.writerow({
                "X": int((x + MARGIN * 2) * SCALE),
                "Y": int((y + MARGIN * 2) * SCALE),
                "Z": int(z * SCALE),
                "ROTATIONS": 63,
                "COPIES": 1
            })

    with open("bins.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["X", "Y", "Z"])
        writer.writeheader()
        writer.writerow({
            "X": int(length * SCALE),
            "Y": int(width * SCALE),
            "Z": int(layer_height * SCALE) + 1
        })

    # parameters.csv: Knapsack objective
    with open("parameters.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["NAME", "VALUE"])
        writer.writeheader()
        writer.writerow({"NAME": "objective", "VALUE": "knapsack"})


    subprocess.run([
        "./packingsolver/build/src/box/packingsolver_box",
        "--items", "items.csv",
        "--verbosity-level", "1",
        "--bins", "bins.csv",
        "--parameters", "parameters.csv",
        "--certificate", "solutions.csv",
        "--time-limit", "5",
        "--objective", "Knapsack",
        "--optimization-mode", "NotAnytimeSequential",
        "--use-sequential-single-knapsack", "true",
        "--use-tree-search", "false",
        "--use-sequential-value-correction", "false",
        "--use-column-generation", "false",
        "--use-dichotomic-search", "false"
    ], capture_output=True, text=True)

    placed_rows = []
    with open("solutions.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["TYPE"] == "ITEM":
                placed_rows.append(row)
    return placed_rows


def box_solution(data, model, boxes, placed_boxes):
    target_space_id = model.body("target_space").id
    target_pos = data.xpos[target_space_id]
    geom_id = model.body_geomadr[target_space_id]
    size = model.geom_size[geom_id]
    length = size[0] * 2
    width = size[1] * 2

    csv_box = []
    for body_id in boxes:
        geom_id = model.body_geomadr[body_id]
        box_size = model.geom_size[geom_id] * 2
        csv_box.append(tuple(box_size))

    origin_x = target_pos[0] + length / 2
    origin_y = target_pos[1] + width / 2
    origin_z = target_pos[2] + 0.001

    all_results = []
    remaining_boxes = csv_box.copy()
    remaining_indices = list(range(len(csv_box)))  
    current_z_offset = 0.0
    level = 0

    while remaining_boxes and level < MAX_LEVELS:

        layer_height = max(box[2] for box in remaining_boxes)

        placed_rows = run_packing_solver(remaining_boxes, length, width, layer_height)

        if not placed_rows:
            break


        placed_solver_ids = []

        for row in placed_rows:
            solver_id = int(row["ID"])
            original_id = remaining_indices[solver_id]

            solver_x = int(row["X"]) / SCALE
            solver_y = int(row["Y"]) / SCALE
            solver_z = int(row["Z"]) / SCALE

            solver_lx = int(row["LX"]) / SCALE - MARGIN * 2
            solver_ly = int(row["LY"]) / SCALE - MARGIN * 2
            solver_lz = int(row["LZ"]) / SCALE

            x_local = solver_x + solver_lx / 2 + MARGIN
            y_local = solver_y + solver_ly / 2 + MARGIN
            z_local = solver_z + solver_lz / 2 + current_z_offset 

            world_x = origin_x - x_local
            world_y = origin_y - y_local
            world_z = origin_z + z_local

            rotation = int(row.get("ROTATION", 0))

            all_results.append({
                "id":       original_id,
                "x":        world_x,
                "y":        world_y,
                "z":        world_z,
                "rotation": rotation
            })
            placed_solver_ids.append(solver_id)

        placed_solver_ids.sort(reverse=True)
        for idx in placed_solver_ids:
            del remaining_boxes[idx]
            del remaining_indices[idx]

        current_z_offset += layer_height
        level += 1

    if remaining_boxes:
        print(f"Warning: {len(remaining_boxes)} boxes could not be placed")


    total_box_volume = 0
    for box in csv_box:
        if tuple(box) not in [csv_box[i] for i in remaining_indices]:
            total_box_volume += box[0] * box[1] * box[2]

    actual_bin_volume = length * width * current_z_offset if current_z_offset > 0 else 1
    packing_efficiency = total_box_volume / actual_bin_volume
    print(f"Packing efficiency: {packing_efficiency*100:.1f}%")

    with open("packing_efficiency.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["packing_efficiency"])
        writer.writerow([round(packing_efficiency * 100, 1)])

    all_results.sort(key=lambda r: r["z"])

    return all_results