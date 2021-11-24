CC = g++

CFLAGS = -std=c++11 -Wall

TMP = ./tmp/

all: $(TMP)main.o $(TMP)database.o
	mkdir $(TMP) -p
	$(CC) $(CFLAGS) -o main.o $(TMP)main.o $(TMP)database.o

debug: CFLAGS += -g
debug: all

$(TMP)main.o: main.cpp
	$(CC) $(CFLAGS) -o $@ -c main.cpp

$(TMP)%.o: %.cpp 
	mkdir $(TMP) -p
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -r $(TMP) -f
	rm main.o -f