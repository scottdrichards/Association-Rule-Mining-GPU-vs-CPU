CC = g++

CFLAGS = -std=c++11 -Wall

TMP = ./tmp/

ALL_FILES = $(basename $(wildcard *.cpp))

all: $(addprefix $(TMP),$(addsuffix .o,$(ALL_FILES)))
	mkdir $(TMP) -p
	$(CC) $(CFLAGS) -o main.out $^

debug: CFLAGS += -g
debug: all

$(TMP)main.o: main.cpp
	mkdir $(TMP) -p
	$(CC) $(CFLAGS) -o $@ -c main.cpp

$(TMP)%.o: %.cpp 
	mkdir $(TMP) -p
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -r $(TMP) -f
	rm main.o -f