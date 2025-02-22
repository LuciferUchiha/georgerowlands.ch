import argparse
import time
import os
import re
import watchdog.events
import watchdog.observers

from nextra_exporter import NextraExporter

def render_notebook(notebook_path: str):
    os.system(f"jupyter nbconvert --to nextra_exporter.NextraExporter {notebook_path}")
    
    # replace all pairs of $$ with ```math and ``` to align with Nextra MDX math block
    with open(notebook_path.replace(".ipynb", ".mdx"), "r") as f:
        def replacer(match):
            # switch between ```math and ``` for each opening and closing set of $$
            replacer.is_opening = not replacer.is_opening
            return "```math" if replacer.is_opening else "```"

        replacer.is_opening = False  # initial state: next $$ becomes ```math

        # Replace all occurrences of $$ using the replacer function
        content = f.read()
        content = re.sub(r'\$\$', replacer, content)
                
    with open(notebook_path.replace(".ipynb", ".mdx"), "w") as f:
        f.write(content)

class Handler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self):
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=['*.ipynb'],
                                                             ignore_directories=True, case_sensitive=False)

    def on_any_event(self, event):
        if not event.src_path.endswith(".ipynb.tmp"):
            print(f"Watchdog received {event.event_type} event - {event.src_path}")
            render_notebook(event.src_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="Path to the content folder that contains the notebooks to be watched and rendered")
    args = parser.parse_args()
    if args.path is None:
        raise ValueError("Path argument is required such as --path ../../content/")

    pages_path = args.path
    # check if path exists
    if not os.path.exists(pages_path):
        raise FileNotFoundError(f"Path {pages_path} does not exist")
    
    # render all notebooks in the pages folder by walking the directory
    print(f"Rendering notebooks in {pages_path}")
    for root, dirs, files in os.walk(pages_path):
        for file in files:
            if file.endswith(".ipynb"):
                render_notebook(os.path.join(root, file))

    # watch for changes in the pages folder and render the notebooks
    print(f"Watching for changes in {pages_path}")
    event_handler = Handler()
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=pages_path, recursive=True)
    observer.start()
    try:
        while True:
            # keep watching
            time.sleep(50)  # not 1 otherwise move events double trigger
    except KeyboardInterrupt:
        observer.stop()
    observer.join()