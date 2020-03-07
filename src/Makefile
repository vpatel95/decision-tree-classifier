CC=g++
CPP_VERSION=-std=c++11
FLAGS=-c -w -O3

all:: classifier

clean::
	@rm -rf *.o classifier

.PHONY: all clean

classifier: classifier.o decision_tree.o
	$(CC) $(CPP_VERSION) classifier.o decision_tree.o -o classifier

classifier.o: classifier.cpp
	$(CC) $(CPP_VERSION) $(FLAGS) classifier.cpp

decision_tree.o: decision_tree.cpp
	$(CC) $(CPP_VERSION) $(FLAGS) decision_tree.cpp

