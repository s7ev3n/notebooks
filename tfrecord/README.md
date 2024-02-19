# tfrecord without tensorflow

## intro

tfrecord is introduced in tensorflow by google. I think this data format is good for data loading, especially when training with large large large amount of text file. 

## installation

### install compiler
you need to install protobuf compiler to compile a `.proto` file:

1) on mac, `brew install protobuf`, and check installation by `protoc --version`
2) on ubuntu, `apt install -y protobuf-compiler`, and check by `protoc --version`

### install runtime
you will like to use some apis provided for protobuf object, install it by `pip install protobuf`.

see some [example](https://protobuf.dev/getting-started/pythontutorial/) and [api reference](https://googleapis.dev/python/protobuf/latest/).


## compile `.proto` file

write the `.proto` file, and compile them using `protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/your_filename.proto`.
you will see `your_filename_pb2.py`.

after getting the `your_filename_pb2.py`, you could distribute it to github, no protoc compiler required to use it on another machine.

## usage

just import the `your_filename_pb2` and use it like a python class, if you want to use some convient function, e.g. ParseDict,
install the runtime to parse it.
