# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: tf_record.proto
# Protobuf Python Version: 4.25.3
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x0ftf_record.proto\x12\x14notebooks.tensorflow\"\x1a\n\tBytesList\x12\r\n\x05value\x18\x01 \x03(\x0c\"\x1e\n\tFloatList\x12\x11\n\x05value\x18\x01 \x03(\x02\x42\x02\x10\x01\"\x1e\n\tInt64List\x12\x11\n\x05value\x18\x01 \x03(\x03\x42\x02\x10\x01\"\xb6\x01\n\x07\x46\x65\x61ture\x12\x35\n\nbytes_list\x18\x01 \x01(\x0b\x32\x1f.notebooks.tensorflow.BytesListH\x00\x12\x35\n\nfloat_list\x18\x02 \x01(\x0b\x32\x1f.notebooks.tensorflow.FloatListH\x00\x12\x35\n\nint64_list\x18\x03 \x01(\x0b\x32\x1f.notebooks.tensorflow.Int64ListH\x00\x42\x06\n\x04kind\"\x97\x01\n\x08\x46\x65\x61tures\x12<\n\x07\x66\x65\x61ture\x18\x01 \x03(\x0b\x32+.notebooks.tensorflow.Features.FeatureEntry\x1aM\n\x0c\x46\x65\x61tureEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12,\n\x05value\x18\x02 \x01(\x0b\x32\x1d.notebooks.tensorflow.Feature:\x02\x38\x01\";\n\x07\x45xample\x12\x30\n\x08\x66\x65\x61tures\x18\x01 \x01(\x0b\x32\x1e.notebooks.tensorflow.FeaturesB\x03\xf8\x01\x01\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tf_record_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  _globals['DESCRIPTOR']._options = None
  _globals['DESCRIPTOR']._serialized_options = b'\370\001\001'
  _globals['_FLOATLIST'].fields_by_name['value']._options = None
  _globals['_FLOATLIST'].fields_by_name['value']._serialized_options = b'\020\001'
  _globals['_INT64LIST'].fields_by_name['value']._options = None
  _globals['_INT64LIST'].fields_by_name['value']._serialized_options = b'\020\001'
  _globals['_FEATURES_FEATUREENTRY']._options = None
  _globals['_FEATURES_FEATUREENTRY']._serialized_options = b'8\001'
  _globals['_BYTESLIST']._serialized_start=41
  _globals['_BYTESLIST']._serialized_end=67
  _globals['_FLOATLIST']._serialized_start=69
  _globals['_FLOATLIST']._serialized_end=99
  _globals['_INT64LIST']._serialized_start=101
  _globals['_INT64LIST']._serialized_end=131
  _globals['_FEATURE']._serialized_start=134
  _globals['_FEATURE']._serialized_end=316
  _globals['_FEATURES']._serialized_start=319
  _globals['_FEATURES']._serialized_end=470
  _globals['_FEATURES_FEATUREENTRY']._serialized_start=393
  _globals['_FEATURES_FEATUREENTRY']._serialized_end=470
  _globals['_EXAMPLE']._serialized_start=472
  _globals['_EXAMPLE']._serialized_end=531
# @@protoc_insertion_point(module_scope)