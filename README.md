singlevec
=========

For when you need a `Vec`, but intend to store only one item
most of the time.  In that case, that single item can be stored
on the stack and will fall back to heap storage for multiple items.

Like a `tinyvec::TinyVec<[T; 1]>` that shares methods
in common with both `Vec` and `Option`.

Simple and 100% safe code for a simple use case.
