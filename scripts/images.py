import re
from pathlib import Path


def convert_markdown_images(content):
    """
    Convert markdown image syntax ![alt](src) to Hugo figure shortcode.

    Args:
        content: String content of the file

    Returns:
        Modified content with converted images
    """
    # Pattern to match markdown images: ![alt text](image-path)
    pattern = r'!\[([^\]]*)\]\(([^)]+)\)'

    def replacement(match):
        alt_text = match.group(1).strip()
        src = match.group(2)

        # Add /images/ prefix if not already present
        if not src.startswith('/images/') and not src.startswith('http'):
            src = '/images' + src if src.startswith('/') else '/images/' + src

        # Build the figure shortcode
        lines = ['{{< figure']
        lines.append(f'  src="{src}"')

        if alt_text:  # Only add alt and caption if alt_text is not empty
            lines.append(f'  alt="{alt_text}"')
            lines.append(f'  caption="{alt_text}"')

        lines.append('>}}')

        return '\n'.join(lines)

    return re.sub(pattern, replacement, content)


def convert_mdx_image_components(content):
    """
    Convert MDX <Image> components to Hugo figure shortcode format.

    Args:
        content: String content of the file

    Returns:
        Modified content with converted Image components to figure shortcodes
    """
    # Pattern to match MDX <Image> components (single line or multi-line)
    pattern = r'<Image\s+([^>]*)/>'

    def replacement(match):
        attrs_text = match.group(1).strip()

        # Extract attributes with proper quote handling
        def extract_attr(attr_name, text):
            # Try double quotes first
            match = re.search(rf'{attr_name}\s*=\s*"([^"]*)"', text)
            if match:
                return match.group(1)
            # Try single quotes
            match = re.search(rf"{attr_name}\s*=\s*'([^']*)'", text)
        src = extract_attr('src', attrs_text)
        caption = extract_attr('caption', attrs_text)
        alt = extract_attr('alt', attrs_text)
        width = extract_attr('width', attrs_text)

        if not src:
            return match.group(0)

        # Add /images/ prefix if not already present
        if not src.startswith('/images/') and not src.startswith('http'):
            src = '/images' + src if src.startswith('/') else '/images/' + src

        # Strip whitespace from caption and alt
        if not src:
            return match.group(0)

        # Strip whitespace from caption and alt
        if caption:
            caption = caption.strip()
        if alt:
            alt = alt.strip()

        # Build figure shortcode
        lines = ['{{< figure']
        lines.append(f'  src="{src}"')

        if alt:  # Only add if alt has content
            lines.append(f'  alt="{alt}"')
        elif caption:  # Use caption as alt if no alt specified and caption has content
            lines.append(f'  alt="{caption}"')

        if caption:  # Only add if caption has content
            lines.append(f'  caption="{caption}"')

        if width:
            lines.append(f'  width="{width}"')

        lines.append('>}}')

        return '\n'.join(lines)

    return re.sub(pattern, replacement, content, flags=re.DOTALL)


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

        modified_content = convert_markdown_images(original_content)
        modified_content = convert_mdx_image_components(modified_content)

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
