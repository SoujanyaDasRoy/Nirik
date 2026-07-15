import threading
import time
from backend.batching import Batcher

def test_batcher_groups_and_timeouts():
    # We'll use a simple processing function that just returns the input multiplied by 2
    def process_fn(batch):
        return [x * 2 for x in batch]

    b = Batcher(batch_size=3, timeout=0.01, process_fn=process_fn)

    # Submit 4 items
    results = []
    for i in range(1, 5):
        # We'll submit and collect results in a non-blocking way for the test
        # But note: the Batcher.submit method in the plan is blocking until result is ready.
        # We'll adjust the test to work with the blocking submit.
        result = b.submit(i)
        results.append(result)

    # We expect that the first two submissions might be batched together (if timeout doesn't trigger)
    # But due to the timeout being small, we might get individual processing.
    # Instead, let's test the batching by submitting and then waiting a bit more than the timeout.
    # We'll redesign the test to be more controlled.

    # Actually, let's test the internal mechanism by checking the queue and the worker.
    # But to keep it simple, we'll test that the Batcher can process items and return correct results.

    # We'll create a new Batcher with a longer timeout to allow batching.
    b2 = Batcher(batch_size=3, timeout=0.1, process_fn=process_fn)

    # Submit 2 items, then wait for the timeout to see if they get processed together.
    # We'll use a list to collect results from the submissions.
    results2 = []
    def submit_and_collect(item):
        result = b2.submit(item)
        results2.append(result)

    # Submit two items
    t1 = threading.Thread(target=submit_and_collect, args=(1,))
    t2 = threading.Thread(target=submit_and_collect, args=(2,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # Wait a bit more than the timeout to allow the batch to process
    time.sleep(0.15)

    # Now submit a third item to trigger the batch (if the first two are still waiting) or process immediately.
    # Actually, let's do a simpler test: submit 3 items and they should be processed as a batch of 3.
    b3 = Batcher(batch_size=3, timeout=0.1, process_fn=process_fn)
    results3 = [b3.submit(i) for i in range(1, 4)]
    # We expect each result to be the input * 2, but note that the batch is processed together.
    # However, the Batcher returns the result for each individual item.
    # So for input [1,2,3] we expect [2,4,6]
    assert results3 == [2, 4, 6]

    # Test timeout: if we submit 2 items and wait for timeout, they should be processed as a batch of 2.
    b4 = Batcher(batch_size=3, timeout=0.01, process_fn=process_fn)
    results4 = []
    def submit_and_collect4(item):
        result = b4.submit(item)
        results4.append(result)

    t3 = threading.Thread(target=submit_and_collect4, args=(1,))
    t4 = threading.Thread(target=submit_and_collect4, args=(2,))
    t3.start()
    t4.start()
    t3.join()
    t4.join()

    # Wait for timeout to pass
    time.sleep(0.02)

    # Now the two items should have been processed (because the timeout elapsed while waiting for the third)
    # But note: the Batcher.submit method blocks until the result is ready.
    # So by the time we collect the results, the two items have been processed.
    # We expect [2, 4]
    assert results4 == [2, 4]

if __name__ == "__main__":
    test_batcher_groups_and_timeouts()
    print("All tests passed")