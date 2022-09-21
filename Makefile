# Copyright (c) 2022 Tavian Barnes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

ALL := ray_box_baseline ray_box_exclusive ray_box_inclusive

CC := clang
CFLAGS := -Wall -O3 -flto -march=native

SIMD ?= 0

all: $(ALL)
.PHONY: all

$(ALL): ray_box_%: black_box.o ray_box_%.o
	+$(CC) $(CFLAGS) $^ -o $@

$(ALL:%=%.o): ray_box_%.o: ray_box.c
	$(CC) $(CFLAGS) -D$(shell impl=$*; echo $${impl^^}) -DSIMD=$(SIMD) -c $< -o $@

black_box.o: black_box.c
	$(CC) -c $< -o $@

check: $(ALL:%=check_%)
.PHONY: check

$(ALL:%=check_%): check_%: %
	./$< check
.PHONY: $(ALL:%=check_%)

bench: all
	./bench.sh
.PHONY: bench

clean:
	$(RM) *.o
.PHONY: clean
