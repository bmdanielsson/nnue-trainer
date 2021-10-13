# Default configuration
variant = release

# Common flags
ARCH += -m64
CFLAGS += -m64 -DIS_64BIT
LDFLAGS += -m64

# Update flags based on build variant
.PHONY : variant
ifeq ($(variant), release)
    CPPFLAGS += -DNDEBUG
    CFLAGS += -O3 -funroll-loops -fomit-frame-pointer $(EXTRACFLAGS)
    LDFLAGS += $(EXTRALDFLAGS)
else
ifeq ($(variant), debug)
    CFLAGS += -g
else
ifeq ($(variant), profile)
    CPPFLAGS += -DNDEBUG
    CFLAGS += -g -pg -O2 -funroll-loops
    LDFLAGS += -pg
endif
endif
endif

# Set special flags needed for different operating systems
ifeq ($(OS), Windows_NT)
    CFLAGS += -DWINDOWS -D_CRT_SECURE_NO_DEPRECATE
    EXEFILE = testdataloader.exe
    LIBFILE = libdataloader.dll
else
ifeq ($(variant), release)
    CFLAGS += -flto
    LDFLAGS += -flto
endif
    LDFLAGS += -lpthread -lm
    EXEFILE = testdataloader
    LIBFILE = libdataloader.so
endif

# Configure warnings
CFLAGS += -W -Wall -Werror -Wno-array-bounds -Wno-pointer-to-int-cast -Wno-int-to-pointer-cast

# Compiler
CC = clang

# Sources
SOURCES = dataloader/board.c \
          dataloader/dataloader.c \
          dataloader/sfen.c \
          dataloader/stream.c \
          dataloader/utils.c

# Intermediate files
OBJECTS = $(SOURCES:%.c=%.o)
DEPS = $(SOURCES:%.c=%.d)
INTERMEDIATES = $(OBJECTS) $(DEPS)

# Include depencies
-include $(SOURCES:.c=.d)

# Targets
.DEFAULT_GOAL = libdataloader

%.o : %.c
	$(COMPILE.c) -MD -o $@ $<

clean :
	rm -f $(EXEFILE) $(LIBFILE) $(INTERMEDIATES)
.PHONY : clean

libdataloader :  $(OBJECTS)
	$(CC) -shared -fpic $(OBJECTS) $(LDFLAGS) -o $(LIBFILE)

testdataloader : $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $(EXEFILE)
