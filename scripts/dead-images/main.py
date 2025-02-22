import os
import re
from pathlib import Path

# Define base project path
PROJECT_ROOT = Path("../../").resolve()
pages_folder = PROJECT_ROOT / "content"
image_folder = PROJECT_ROOT / "public"

def normalize_path(path: str, base_path: Path) -> str:
    """
    Normalize a path to be relative to the base path
    """
    try:
        full_path = Path(path).resolve()
        return str(full_path.relative_to(base_path))
    except ValueError:
        return str(path)

def find_md_images(mdx_files: list[str], image_files: list[str]) -> list[str]:
    """
    Find all markdown images in mdx files, check if they exist and return the unused ones
    """
    used_images = []
    for mdx_file in mdx_files:
        with open(mdx_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line_num, line in enumerate(lines, 1):
                match = re.search(r'!\[.*\]\((.*)\)', line)
                if match:
                    image_path = match.group(1)
                    full_image_path = Path(os.path.dirname(mdx_file)) / image_path
                    normalized_path = normalize_path(full_image_path, PROJECT_ROOT)
                    used_images.append(normalized_path)
                    
                    if normalized_path not in image_files:
                        print(f"Image {normalized_path} not found in {mdx_file} on line {line_num}")
    
    return list(set(image_files) - set(used_images))

def find_next_images(mdx_files: list[str], image_files: list[str]) -> list[str]:
    """
    Find all Next.js Image components in mdx files
    """
    used_images = []
    for mdx_file in mdx_files:
        with open(mdx_file, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line_num, line in enumerate(lines, 1):
                match = re.search(r'\<Image.*src="(.*)".*\/\>', line)
                if match:
                    image_path = match.group(1)
                    full_image_path = Path(os.path.dirname(mdx_file)) / image_path
                    normalized_path = normalize_path(full_image_path, PROJECT_ROOT)
                    used_images.append(normalized_path)

                    if normalized_path not in image_files:
                        print(f"Image {normalized_path} not found in {mdx_file} on line {line_num}")
                    
    return list(set(image_files) - set(used_images))

if __name__ == '__main__':
    # Collect all image files and normalize their paths
    image_files = []
    for root, _, files in os.walk(image_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                full_path = Path(root) / file
                normalized_path = normalize_path(full_path, PROJECT_ROOT)
                image_files.append(normalized_path)
    
    # Collect all MDX files
    mdx_files = []
    for root, _, files in os.walk(pages_folder):
        for file in files:
            if file.endswith('.mdx'):
                mdx_files.append(str(Path(root) / file))

    unused_md_images = find_md_images(mdx_files, image_files)
    unused_next_images = find_next_images(mdx_files, image_files)
    
    unused_images = set(unused_md_images).intersection(unused_next_images)
    
    for image in unused_images:
        print(f"Unused image: {image}")




