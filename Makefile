.PHONY: all test run clean

all:
	$(MAKE) -C amr-gpu-ml all

test:
	$(MAKE) -C amr-gpu-ml test

run:
	$(MAKE) -C amr-gpu-ml run

clean:
	$(MAKE) -C amr-gpu-ml clean
