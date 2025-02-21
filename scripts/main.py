
# for all files in ../content/ that end with .mdx replace all pairs of $$ with ```math and ``` to align with Nextra MDX math block
import os
import re

content_path = "../content/"
for root, dirs, files in os.walk(content_path):
    for file in files:
        if file.endswith(".mdx"):
            with open(os.path.join(root, file), "r") as f:
                def replacer(match):
                    # switch between ```math and ``` for each opening and closing set of $$
                    replacer.is_opening = not replacer.is_opening
                    return "```math" if replacer.is_opening else "```"

                replacer.is_opening = False  # initial state: next $$ becomes ```math

                # Replace all occurrences of $$ using the replacer function
                content = f.read()
                content = re.sub(r'\$\$', replacer, content)
                        
            with open(os.path.join(root, file), "w") as f:
                f.write(content)