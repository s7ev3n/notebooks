import os
import shutil
import tempfile
import multiprocessing
from record import RecordWriter
from reader import DatasetReader


if __name__ == '__main__':
    # Firstly, write data to the records.
    features = {'a': ['float'], 'b': ['int'], 'c': ['bytes'], 'd': 'string'}
    data1 = {'a': [1., 2., 3.], 'b': [4, 5, 6], 'c': [b'7', b'8', b'9'], 'd': '1'}
    data2 = {'a': [2., 3., 4.], 'b': [5, 6, 7], 'c': [b'8', b'9', b'10'], 'd': '2'}
    path_to_records = os.path.join(tempfile.gettempdir(), 'my_records')
    
    if os.path.exists(path_to_records):
        shutil.rmtree(path_to_records)
    os.makedirs(path_to_records)
    with RecordWriter(path_to_records, features) as writer:
        writer.write(data1)
        writer.write(data2)

    # Next, create a prefetching queue.
    batch_size = 64
    output_queue = multiprocessing.Queue(batch_size)

    # Finally, create and start a dataset reader.
    dataset_reader = DatasetReader(path_to_records, output_queue)
    dataset_reader.start()

    # Enjoy the training loop.
    for i in range(10):
        data = output_queue.get()
        print(type(data))
        print(data)
