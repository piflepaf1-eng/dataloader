The following are ideas categorized by type: should have, could have (it should be clarified if it is desireable from a technical or DX standpoint), and nice to have.

# Should have
- None at the moment. Feel free to suggest !

# Could have
- leverage external libraries to increase performance or reduce loc of this library, maybe: ZMQ
- write performance critical code in a faster language, eg C++, Rust, etc. (this is a big one, I'm not sure if it's worth it)
- review the organisation of the project. Are files properly named ? Are the correct levels of abstraction used ? Does the documentation really reflects the functionality ? One example I'm thinking about: there could be a "sources" module that would contain mixing source (renamed for the occasion to something more precise) and wds_source (whose name is clear). Backends "cpu" may be better named as "torchvision" maybe ? Also, the current "mixing source" could be a submodule of sources, containing all the classes for mixing sources (I believe shard cache would be here ?)

# Nice to have
- add support for other data sources, S3 in particular. Maybe to be done in dino_datasets ? Not useful at the moment anyway.
- expose augmentations for use in other libraries, eg torchvision (collate_fn and/or transform ?) to be discussed.