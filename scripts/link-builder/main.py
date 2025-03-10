import os
import json
import re

if __name__ == "__main__":
    # checks that all links in the digital garden are valid
    # and creates a graph of the digital garden

    pages_path = "../../content/"

    nodes = []
    links = []

    # walk through the entire digital garden to find all the nodes
    for root, dirs, files in os.walk(pages_path):
        for file in files:
            if file.endswith(".mdx"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    print(f"Node for {os.path.join(root, file)}")
                    content = f.read()
                    # relative unix style path, no backslashes
                    url = "/" + \
                        os.path.relpath(os.path.join(root, file),
                                        pages_path).replace("\\", "/")
                    # remove the extension
                    url = url.replace(".mdx", "")

                    # if the file is index then the url should be the directory
                    if url.endswith("index"):
                        url = url[:-5]

                    for line in content.split("\n"):
                        # if it is a title make a node
                        if line.startswith("# "):
                            title = line[2:]

                            nodes.append({"id": url, "name": title})

    # walk through the entire digital garden to find all the links
    for root, dirs, files in os.walk(pages_path):
        for file in files:
            if file.endswith(".mdx"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    print(f"Link for {os.path.join(root, file)}")
                    content = f.read()
                    # relative unix style path, no backslashes
                    url = "/" + \
                        os.path.relpath(os.path.join(root, file),
                                        pages_path).replace("\\", "/")
                    # remove the extension
                    url = url.replace(".mdx", "")
                    for node in nodes:
                        # either the node id is one of these two patterns:
                        # - (node["id"])
                        # - (node["id"]#some-text)
                        if re.search(f"\\({node['id']}\\)", content) or re.search(f"\\({node['id']}#.*\\)", content):
                            links.append({"source": url, "target": node["id"]})

    print(f"Nodes: {len(nodes)}")
    print(f"Links: {len(links)}")

    # save the nodes and links to a json file
    print("Exporting graph.json")
    with open("../data/graph.json", "w") as f:
        json.dump({"nodes": nodes, "links": links}, f,
                  sort_keys=True, indent=4, separators=(',', ': '))

    print("Completed graph-builder")
