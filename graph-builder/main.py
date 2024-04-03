import os
import json


if __name__ == "__main__":
    print("Starting graph-builder")

    pages_path = "../pages/"

    nodes = []
    links = []

    # walk through the entire digital garden
    for root, dirs, files in os.walk(pages_path):
        for file in files:
            if file.endswith(".mdx"):
                with open(os.path.join(root, file), "r") as f:
                    print(f"Reading {os.path.join(root, file)}")
                    content = f.read()
                    for line in content.split("\n"):
                        # if it is a title make a node
                        if line.startswith("# "):
                            title = line[2:]
                            nodes.append({"id": title, "name": title})
                        # TODO do more then just the titles and make links


    # TODO for now just make links between all nodes
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            links.append({"source": nodes[i]["id"], "target": nodes[j]["id"]})

    # save the nodes and links to a json file
    print("Exporting graph.json")
    with open("../data/graph.json", "w") as f:
        json.dump({"nodes": nodes, "links": links}, f)

    print("Completed graph-builder")