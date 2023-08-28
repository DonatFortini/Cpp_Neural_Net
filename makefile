CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11

# The name of your executable
TARGET = neuralNet

# Source files
SRCS = neuralNet.cpp matrix.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Dependencies
DEPS = neuralNet.hpp matrix.hpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ -lm
	./$(TARGET)

%.o: %.cpp $(DEPS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGET)
