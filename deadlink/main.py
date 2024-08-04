import os
import re

pages_folder = "../pages"


def check_for_md_images(mdx_files: list[str]):
    """
    Check for markdown images in mdx files
    """
    num_missing_images = 0
    for mdx_file in mdx_files:
        with open(mdx_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                # check for markdown image notation
                match = re.search(r'!\[.*\]\((.*)\)', line)
                if match:
                    print(f"Found markdown image notation in {
                          mdx_file} on line {lines.index(line) + 1}")

                    # check if the image file exists
                    image_path = os.path.join(
                        os.path.dirname(mdx_file), match.group(1))
                    if not os.path.exists(image_path):
                        print(f"Image file {image_path} does not exist")
                        num_missing_images += 1

    return num_missing_images


def check_for_next_images(mdx_files: list[str]):
    """
    Check for Next images in mdx files
    """
    pass


def check_page_links(mdx_files: list[str]):
    """
    Check for page links in mdx files
    """
    num_dead_links = 0
    for mdx_file in mdx_files:
        with open(mdx_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in lines:
                # check for page links, make sure not an image link so no exclamation mark in front with negative lookbehind
                match = re.search(r'\[.*\]\((?!.*\..*\))(.*)\)', line)

                # TODO handle in between ``` code blocks and ` inline code blocks
                if match:
                    link = match.group(1)
                    # if it's a local link, i.e to a header in the same file it starts with a hash
                    if link.startswith("#"):
                        continue

                    page_path = os.path.join(
                        os.path.dirname(mdx_file), link)

                    # remove the "header" part of the path, i.e the hash part
                    page_path = page_path.split("#")[0]

                    # full path to the page file
                    page_path = f"{pages_folder}{page_path}.mdx"
                    print(f"Found page link in {mdx_file} on line {
                        lines.index(line) + 1} to {page_path}")

                    if not os.path.exists(page_path):
                        print(f"Page link from {mdx_file} on line {
                              lines.index(line) + 1} to {page_path} does not exist")
                        num_dead_links += 1

    return num_dead_links


if __name__ == '__main__':
    mdx_files = []

    for root, dirs, files in os.walk(pages_folder):
        for file in files:
            if file.endswith(".mdx"):
                mdx_files.append(os.path.join(root, file).replace("\\", "/"))

    print(f"----- Checking for markdown images in mdx files -----")
    missing_images = check_for_md_images(mdx_files)
    print(f"----- Checking for Next images in mdx files -----")
    check_for_next_images(mdx_files)
    print(f"----- Checking for page links in mdx files -----")
    dead_links = check_page_links(mdx_files)
    print(f"Total missing images: {missing_images}")
    print(f"Total dead links: {dead_links}")
