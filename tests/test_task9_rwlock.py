import threading
import time

from picovdb.pico_vdb import _RWLock


def test_rwlock_readers_coexist_writer_excludes():
    lock = _RWLock()
    max_readers = 0
    active_readers = 0
    writer_active = False
    errors = []

    def reader():
        nonlocal active_readers, max_readers, writer_active
        with lock.read_lock():
            if writer_active:
                errors.append("Reader overlapped with writer")
            active_readers += 1
            max_readers = max(max_readers, active_readers)
            time.sleep(0.05)
            active_readers -= 1

    def writer():
        nonlocal writer_active, active_readers
        with lock.write_lock():
            writer_active = True
            if active_readers != 0:
                errors.append("Writer overlapped with readers")
            time.sleep(0.05)
            writer_active = False

    # Start two readers nearly simultaneously
    t1 = threading.Thread(target=reader)
    t2 = threading.Thread(target=reader)
    t1.start()
    t2.start()
    time.sleep(0.01)
    # Start writer which should wait for readers
    tw = threading.Thread(target=writer)
    tw.start()
    t1.join()
    t2.join()
    tw.join()

    assert max_readers >= 2
    assert not errors
