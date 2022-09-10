To execute the benchmark, simply run

```console
$ make bench
```

The results in the [blog post](https://tavianator.com/2022/ray_box_boundary.html) were produced with

```console
$ make bench CONST_HZ=1
```

to disable CPU frequency scaling, but that may not be as portable.
