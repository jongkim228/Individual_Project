import mujoco
import csv
import subprocess

SCALE = 1000


def box_packing(data,model,boxes):
    target_space_id = model.body("target_space").id
    geom_id = model.body_geomadr[target_space_id]

    size = model.geom_size[geom_id]

    print("target_space raw size:", size)
    print("length:", size[0] * 2)
    print("width:", size[1] * 2)

    length = size[0] * 2
    width = size[1] * 2
    height = 0.7

    area = length * width
    dimension = area * height

    box_in_space = 0
    packing_ratio = (box_in_space / dimension) * 100

    csv_box = []

    for body_id in boxes:
        geom_id = model.body_geomadr[body_id]
        box_size = model.geom_size[geom_id]        
        pos = data.geom_xpos[geom_id]

        csv_box.append(box_size)

    
    for i in csv_box:
        print(i)


    with open("items.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["X", "Y", "Z", "ROTATIONS", "COPIES"])
        writer.writeheader()                          
        for x, y, z in csv_box:
            writer.writerow({
                "X": int(x * 2 * SCALE), "Y": int(y * 2 * SCALE),"Z": int(z * 2 * SCALE),
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
            "NAME": "objective", "VALUE": "knapsack"
        })


    subprocess.run([
    "./packingsolver/build/src/box/packingsolver_box",
    "--items", "items.csv",
    "--bins", "bins.csv",
    "--parameters", "parameters.csv",
    "--certificate", "solutions.csv",
    "--time-limit", "10"
])

    

    results = []
    with open("solutions.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["TYPE"] == "ITEM":
                results.append(
                    {
                        "id": int(row["ID"]),
                        "x": int(row["X"]) / SCALE,
                        "y": int(row["Y"]) / SCALE,
                        "z": int(row["Z"]) / SCALE,
                    }
                )

    return results

