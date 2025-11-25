CC := gcc
CFLAGS := -O3 -Wall -Wextra -march=native -ffast-math
LDFLAGS := -pthread -lm

TARGET := pi.out
SRCS := pi.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
