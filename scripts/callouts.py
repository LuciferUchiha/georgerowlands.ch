import re
from pathlib import Path


def replace_callout_blocks(content):
    """
    Replace JSX-style <Callout> blocks with Hugo shortcode {{< callout >}} notation.

    Handles both:
    - <Callout type="example"> ... </Callout>
    - <Callout type="info" title="Title"> ... </Callout>

    Args:
        content: String content of the file

    Returns:
        Modified content with replaced callout blocks
    """
    pattern = r'<Callout\s+type="([^"]+)"(?:\s+title="([^"]+)")?\s*>\s*\n(.*?)</Callout>'

    def replacement(match):
        callout_type = match.group(1)
        title = match.group(2)
        content = match.group(3).rstrip('\n')

        if title:
            opening = f'{{{{< callout type="{callout_type}" title="{title}" >}}}}'
        else:
            opening = f'{{{{< callout type="{callout_type}" >}}}}'

        closing = '{{< /callout >}}'

        return f'{opening}\n{content}\n{closing}'

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

    md_files = list(content_path.rglob("*.md"))

    if not md_files:
        print("No markdown files found")
        return

    print(f"Found {len(md_files)} markdown files")
    modified_count = 0

    for md_file in md_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            original_content = f.read()

        modified_content = replace_callout_blocks(original_content)

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
