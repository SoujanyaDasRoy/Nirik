import threading
import time
from queue import Queue, Empty

class Batcher:
    def __init__(self, batch_size, timeout, process_fn):
        """
        :param batch_size: maximum number of items to process in a batch
        :param timeout: maximum time to wait for the batch to fill (in seconds)
        :param process_fn: function to process a batch of items, should take a list and return a list of results
        """
        self.batch_size = batch_size
        self.timeout = timeout
        self.process_fn = process_fn
        self.queue = Queue()
        self.results = {}
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            batch = []
            batch_ids = []
            try:
                # Get first item with timeout
                item_id, item = self.queue.get(timeout=self.timeout)
                batch.append(item)
                batch_ids.append(item_id)
                # Fill up to batch_size without blocking
                while len(batch) < self.batch_size:
                    try:
                        item_id, item = self.queue.get_nowait()
                        batch.append(item)
                        batch_ids.append(item_id)
                    except Empty:
                        break
                # Process batch
                processed = self.process_fn(batch)
                for idx, res in zip(batch_ids, processed):
                    with self.lock:
                        self.results[item_id] = res
            except Empty:
                continue

    def submit(self, item):
        item_id = id(item)
        self.queue.put((item_id, item))
        # Wait for result (simplistic)
        while True:
            with self.lock:
                if item_id in self.results:
                    return self.results.pop(item_id)
            time.sleep(0.001)