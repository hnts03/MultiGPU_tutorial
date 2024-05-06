CC = nvcc
CFLAGS = -g -Wall

all-ignore-cpu: single-nostream multi-nostream single-stream multi-stream

all: cpu single-nostream multi-nostream single-stream multi-stream

cpu : main.cu
	$(CC) $(CCFLAGS) -DCPU=1 -o $@ $^

single-nostream : main.cu
	$(CC) $(CCFLAGS) -DGPU=1 -DSTREAM_ENABLE=0 -DMULTI=0 -o $@ $^

single-nostream-debug : main.cu
	$(CC) $(CCFLAGS) -DDEBUG=1 -DGPU=1 -DSTREAM_ENABLE=0 -DMULTI=0 -o $@ $^

multi-nostream : main.cu
	$(CC) $(CCFLAGS) -DGPU=1 -DSTREAM_ENABLE=0 -DMULTI=1 -o $@ $^

multi-nostream-debug : main.cu
	$(CC) $(CCFLAGS) -DDEBUG=1 -DGPU=1 -DSTREAM_ENABLE=0 -DMULTI=1 -o $@ $^

single-stream : main.cu
	$(CC) $(CCFLAGS) -DGPU=1 -DSTREAM_ENABLE=1 -DMULTI=0 -o $@ $^

single-stream-debug : main.cu
	$(CC) $(CCFLAGS) -DDEBUG=1 -DGPU=1 -DSTREAM_ENABLE=1 -DMULTI=0 -o $@ $^

multi-stream : main.cu
	$(CC) $(CCFLAGS) -DGPU=1 -DSTREAM_ENABLE=1 -DMULTI=1 -o $@ $^

multi-stream-debug : main.cu
	$(CC) $(CCFLAGS) -DDEBUG=1 -DGPU=1 -DSTREAM_ENABLE=1 -DMULTI=1 -o $@ $^



clean : 
	rm -f cpu
	rm -f single-nostream
	rm -f single-nostream-debug
	rm -f multi-nostream
	rm -f multi-nostream-debug
	rm -f single-stream
	rm -f single-stream-debug
	rm -f multi-stream
	rm -f multi-stream-debug
