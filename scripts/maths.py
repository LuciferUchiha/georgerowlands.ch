import re
from pathlib import Path


def replace_math_blocks(content):
    """
    Replace ```math ... ``` blocks with $$ ... $$ notation.

    Args:
        content: String content of the file

    Returns:
        Modified content with replaced math blocks
    """
    # Pattern to match ```math or ```{math} blocks
    pattern = r'```\{?math\}?\n(.*?)```'

    def replacement(match):
        formula = match.group(1).rstrip('\n')
        return f'$$\n{formula}\n$$'

    # Use DOTALL flag to match across multiple lines
    modified_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    return modified_content


def process_markdown_files(content_dir):
    """
    Process all markdown files in the content directory.

    Args:
        content_dir: Path to the content directory
    """
    content_path = Path(content_dir)

    if not content_path.exists():
        print(f"Error: Directory {content_dir} does not exist")
        return

    # Find all markdown files
    md_files = list(content_path.rglob("*.md"))

    if not md_files:
        print("No markdown files found")
        return

    print(f"Found {len(md_files)} markdown files")
    modified_count = 0

    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            original_content = f.read()

        modified_content = replace_math_blocks(original_content)

        if modified_content != original_content:
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            print(f"✓ Modified: {md_file.relative_to(content_path.parent)}")
            modified_count += 1

    print(f"\nCompleted: {modified_count} files modified")


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    content_dir = script_dir.parent / "content"

    print(f"Processing markdown files in: {content_dir}\n")
    process_markdown_files(content_dir)
