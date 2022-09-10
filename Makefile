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

COUNT := 10000000000

all: $(ALL)
.PHONY: all

$(ALL): ray_box_%: black_box.o ray_box_%.o
	+$(CC) $(CFLAGS) $^ -o $@

$(ALL:%=%.o): ray_box_%.o: ray_box.c
	$(CC) $(CFLAGS) -D$(shell impl=$*; echo $${impl^^}) -c $< -o $@

black_box.o: black_box.c
	$(CC) -c $< -o $@

bench: const_hz $(ALL:%=bench_%) unconst_hz
.PHONY: bench

# Don't run the benchmarks in parallel
.NOTPARALLEL:

$(ALL:%=bench_%): bench_%: %
	@printf '%9s (height %2d): ' $(<:ray_box_%=%) 4
	@./$< 4 $(COUNT)
	@printf '%9s (height %2d): ' $(<:ray_box_%=%) 5
	@./$< 5 $(COUNT)
	@printf '%9s (height %2d): ' $(<:ray_box_%=%) 8
	@./$< 8 $(COUNT)
	@printf '%9s (height %2d): ' $(<:ray_box_%=%) 10
	@./$< 10 $(COUNT)
	@printf '---\n'
.PHONY: $(ALL:%=bench_%)

const_hz:
ifdef CONST_HZ
	sudo cpupower frequency-set -g performance >/dev/null
	echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null
endif
	@:
.PHONY: const_hz

unconst_hz:
ifdef CONST_HZ
	sudo cpupower frequency-set -g schedutil >/dev/null
	echo 1 | sudo tee /sys/devices/system/cpu/cpufreq/boost >/dev/null
endif
	@:
.PHONY: unconst_hz

clean:
	$(RM) *.o
.PHONY: clean
