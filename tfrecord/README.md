# tfrecord without tensorflow

## intro

tfrecord is introduced in tensorflow by google. I think this data format is good for data loading, especially when training with large large large amount of text file. 

## installation

you need to install protobuf compiler to compile a `.proto` file:

1) on mac, `brew install protobuf`, and check installation by `protoc --version`
2) on ubuntu, `apt install -y protobuf-compiler`, and check by `protoc --version`

## compile `.proto` file

write the `.proto` file, and compile them using `protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/your_filename.proto`.
you will see `your_filename_pb2.py`.

