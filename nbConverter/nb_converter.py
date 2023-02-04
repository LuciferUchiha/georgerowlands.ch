import time
import os
import watchdog.events
import watchdog.observers


class Handler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self):
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=['*.ipynb'],
                                                             ignore_directories=True, case_sensitive=False)

    def on_any_event(self, event):
        if not event.src_path.endswith(".ipynb.tmp"):
            print(f"Watchdog received {event.event_type} event - {event.src_path}")
            os.system(f"jupyter nbconvert --to markdown {event.src_path}")


if __name__ == "__main__":
    print("Watchdog is watching")
    pages_path = "../pages/"
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
