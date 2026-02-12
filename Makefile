CC ?= cc
CFLAGS ?= -O2 -Wall -Wextra -std=c11

TARGET := microgpt
SRC := microgpt.c

.PHONY: all build run clean

all: build

build: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

run: build
	./$(TARGET)

clean:
	rm -f $(TARGET)
