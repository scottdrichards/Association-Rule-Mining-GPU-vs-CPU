CC = g++

CFLAGS = -std=c++11 -Wall

TMP = ./tmp/

all: $(TMP)main.o $(TMP)database.o $(TMP)progressBar.o $(TMP)dataset.o
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