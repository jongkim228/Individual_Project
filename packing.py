import mujoco
import csv
import subprocess

SCALE = 1000
MARGIN = 0.01


def box_solution(data,model,boxes,placed_boxes):
    target_space_id = model.body("target_space").id
    target_pos = data.xpos[target_space_id]

    geom_id = model.body_geomadr[target_space_id]
    size = model.geom_size[geom_id]
    length = size[0] * 2
    width = size[1] * 2

    print("LENGTH",length)
    print("TARGET_POS", target_pos)
    

    floor_area = length * width
    placed_area = 0
    for i in placed_boxes:
        box_geom_id = model.body_geomadr[i]
        located_box_size = model.geom_size[box_geom_id]
        usage = located_box_size[0] * located_box_size[1] * 4
        placed_area += usage

    area_usage = placed_area / floor_area

    if area_usage >= 0.08:
        height = 0.16
    else:
        height = 0.08

    csv_box = []

    for body_id in boxes:
        geom_id = model.body_geomadr[body_id]
        box_size = model.geom_size[geom_id] * 2     
        csv_box.append(box_size)

    
    for i in csv_box:
        print(i)


    with open("items.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["X", "Y", "Z", "ROTATIONS", "COPIES"])
        writer.writeheader()                          
        for x, y, z in csv_box:
            writer.writerow({
                "X": int((x+MARGIN * 2)  * SCALE), "Y": int((y+MARGIN * 2) * SCALE) ,"Z": int((z*SCALE)),
                "ROTATIONS": 1, 
                "COPIES": 1
            })

    with open("bins.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["X", "Y", "Z"])
        writer.writeheader()  
        writer.writerow({
            "X": int(length * SCALE),
            "Y": int(width * SCALE),
            "Z": int(height * SCALE)
        })


    with open("parameters.csv","w",newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["NAME", "VALUE"])
        writer.writeheader()  
        writer.writerow({
            "NAME": "objective", "VALUE": "bin-packing"
        })


    subprocess.run([
    "./packingsolver/build/src/box/packingsolver_box",
    "--items", "items.csv",
     "--verbosity-level", "0",
    "--bins", "bins.csv",
    "--parameters", "parameters.csv",
    "--certificate", "solutions.csv",
    "--time-limit", "10",
])
    
    with open("solutions.csv", "r") as f:
        print(f.read()) 
    
    origin_x = target_pos[0] - length / 2
    origin_y = target_pos[1] - width / 2
    origin_z = target_pos[2]

    

    results = []
    with open("solutions.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["TYPE"] == "ITEM":
                item_id = int(row["ID"])
                box_size = csv_box[item_id]
                results.append(
                    {
                        "id": item_id,
                        "x": origin_x + int(row["X"]) / SCALE + box_size[0] / 2,
                        "y": origin_y + int(row["Y"]) / SCALE + box_size[1] / 2,
                        "z": origin_z + int(row["Z"]) / SCALE + box_size[2] / 2,
                    }
                )

    for r in results:
        print(f"SOLUTION x={r['x']:.4f} y={r['y']:.4f} z={r['z']:.4f}")

    return results

