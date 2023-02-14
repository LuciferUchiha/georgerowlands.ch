---
title: IPC with Pipes
tags: [c, ipc, processes, pipes]
---

# IPC with Pipes

Interprocess comminucation - IPC is the subject of coomunicatiing, echanging data and synchronizing between to processes.

A kategorie of IPC are Datatransfers which use read and write system calls. The second category commincates via shared memory, without system calls and is therefore also faster.

Datatransfers can be byte streams, message based or use special pseduoterminals. Important with all these mechanisms is that a read operation is destructive. Meaning if data has been read then it is no longer available to the others. Synchronization is done automaticallly. If there is no data available then read operation blocks.

Pipes, FIFOs, Stream sockets are unlimited byte streams which means that the number bytes does not matter.

Message queues and datagram sockets are messaged based. Each read operation reads one message exactly how it was written. It is not possible to only read a part of a message or multiple at once.

Shared Memory and memory mappings are fast but need to be synchronized. reading is however not destructive. Often semaphores are used.

## File locks

Work same as ReadWriteLock in Java.
coordiante file access. Read locks can be shared between multiples however if a process has a write lock then no other thread can have a read or write lock.
flock and fcntl system calls???

## Pipes

A pipe "|" is a form of redirection of standard output to some other destination that can be used in shells on Unix operating systems. So for example you can redirect the stdout of a command to a file or connect it to the stdin of another command. Pipes are unidirectional i.e data flows from left to right through the pipe. The command-line programs (right side) that do the further processing are referred to as filters. You can also use pipes programmatically in C for IPC.

read blocks, if pipe is closed returns 0/EOF.

Just a buffer in or file kernel memory with max capacity of 64KB. If a pipe is full write blocks until on the other end data is read.

pipe puts 2 file descriptors into the passed array the first (index 0) being the read end of the pipe and the other file descriptor for the write end.

when finished with writing need to close it so read gets EOF and doesn't block indefinetly. If the pipe was closed on read side and the process still writes to the pipe the kernel sends a SIGPIPE signal. if the signal is ignored then write returns the error EPIPE.

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void)
{
    int fd[2]; // 0 = read, 1 = write end
    pipe(fd);

    int id = fork();

    if (id == -1) { printf("Error when forking"); return 1; }
    if (id == 0) // child process
    {
        close(fd[0]); // close read end
        int x;
        printf("CHILD: Input a number: ");
        scanf("%d", &x);
        if (write(fd[1], &x, sizeof(x)) == -1) { printf("Error writing to pipe"); return 3; }
        close(fd[1]); // close write end when finished
    }
    else // parent process
    {
        close(fd[1]); // close write end
        int y;
        if (read(fd[0], &y, sizeof(y)) == -1) { printf("Error reading from pipe"); return 3; }
        printf("PARENT: You put in: %d\n", y);
        close(fd[0]); // close read end when finished
    }
    return 0;
}
```

```bash filename="output"
CHILD: Input a number: 4
PARENT: You put in: 4
```

### Bidirectional Pipes

There can be scenarios where you want bidirectional communication between two processes using pipes. This can't be done with just one pipe for this you need two pipes between the processes.

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void)
{
    /* Send a number from parent to child, child processes the number and sends back to parent. */
    const int READ_END = 0; const int WRITE_END = 1;
    int childToParent[2];
    int parentToChild[2];
    if(pipe(childToParent) == -1) { printf("Error creating pipe"); return 1; }
    if(pipe(parentToChild) == -1) { printf("Error creating pipe"); return 1; }

    int id = fork();
    if (id == -1)
    {
        printf("Error when forking");
        return 2;
    }
    if (id == 0) // child process
    {
        close(childToParent[READ_END]);
        close(parentToChild[WRITE_END]);
        int input;
        if (read(parentToChild[READ_END], &input, sizeof(input)) == -1)
        {
            printf("Error when reading from parent to child");
            return 3;
        }
        printf("CHILD: Received: %d\n", input);
        int output = input * 2; // input gets doubled
        if (write(childToParent[WRITE_END], &output, sizeof(output)) == -1)
        {
            printf("Error when writing from child to parent");
            return 4;
        }
        printf("CHILD: Sent: %d\n", output);
    }
    else // parent process
    {
        close(parentToChild[READ_END]);
        close(childToParent[WRITE_END]);
        printf("PARENT: Input a number: ");
        int input;
        scanf("%d", &input);
        if (write(parentToChild[WRITE_END], &input, sizeof(input)) == -1)
        {
            printf("Error when writing from parent to child");
            return 5;
        }
        printf("PARENT: Sent: %d\n", input);
        int output;
        if (read(childToParent[READ_END], &output, sizeof(output)) == -1)
        {
            printf("Error when reading from child to parent");
            return 6;
        }
        printf("PARENT: Received: %d\n", output);
        close(parentToChild[WRITE_END]);
        close(childToParent[READ_END]);
    }
    return 0;
}
```

```bash filename="output"
PARENT: Input a number: 5
PARENT: Sent: 5
CHILD: Received: 5
CHILD: Sent: 10
PARENT: Received: 10
```

### Simulating the "|" Pipe Operator

As you might have expected it is possible to simulate the pipe operator in the shell with pipes in C.

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void)
{
    const int READ_END = 0;
    const int WRITE_END = 1;
    int fd[2];
    if (pipe(fd) == -1)
    {
        printf("Error creating pipe");
        return 1;
    }

    int pingForkId = fork();
    if (pingForkId == -1)
    {
        printf("Error when forking");
        return 2;
    }
    if (pingForkId == 0) // child process
    {
        dup2(fd[WRITE_END], STDOUT_FILENO); // copies pipe write fd to stdout fd
        close(fd[READ_END]);
        close(fd[WRITE_END]); // can be closed as still a reference pointing to it

        execlp("ping", "ping", "-c", "5", "google.com", NULL); // current process code gets replaced by new process code. system() would create another child process.
    }
    int grepForkId = fork();
    if (grepForkId == -1)
    {
        printf("Error when forking");
        return 2;
    }
    if (grepForkId == 0) // child process
    {
        dup2(fd[READ_END], STDIN_FILENO); // copies pipe read fd to stdin fd
        close(fd[READ_END]);
        close(fd[WRITE_END]);

        execlp("grep", "grep", "rtt", NULL);
    }

    close(fd[READ_END]);
    close(fd[WRITE_END]);
    // Wait for children to terminate
    waitpid(pingForkId, NULL, 0);
    waitpid(grepForkId, NULL, 0);
    return 0;
}
```

### Synchronizing with Pipes

As mentioned above pipes don't need to transfer data, they can also just be used for synchronization between processes.

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void)
{
    int fd[2]; /* Process synchronization pipe */

    if (pipe(fd) == -1)
    {
        printf("Error creating pipe");
        return 1;
    }

    for (int i = 0; i < 10; i++)
    {
        switch (fork())
        {
        case -1:
            printf("Error when forking");
            return 2;
        case 0:           // child process
            close(fd[0]); // close read end

            // child does some work and lets parent know when finished
            for (int j = 0; j < 100000000; j++)
            {
            }
            printf("Finished work in Process %d\n", i);
            // notifies parent that done by decrementing file descriptor count
            close(fd[1]);
            exit(EXIT_SUCCESS);
        default:
            break; // parent continue with loop
        }
    }
    printf("Finished creating all children\n");
    close(fd[1]); // parent doesn't use write end
    int dummy;
    // blocks till all are finished and receives EOF
    if (read(fd[0], &dummy, 1) != 0)
    {

        printf("Parent didn't get EOF");
        return 3;
    }
    printf("All finished");
    return 0;
}
```

```bash filename="output"
Finished creating all children
Finished work in Process 0
Finished work in Process 1
Finished work in Process 2
Finished work in Process 4
Finished work in Process 3
Finished work in Process 5
Finished work in Process 6
Finished work in Process 7
Finished work in Process 8
Finished work in Process 9
All finished
```

## FIFO files (Named Pipes)

pipes only work in the same hierarchy so between only between a parent and its child process or between children that share the same parent. We might want to be able to communicate between two processes that are not related. For this we have FIFOs which are a variation of pipes that work very similiarly to files, and also use a file which is why they are also often reffered to as named pipes. FIFOs work the same way as pipes so they also have unidrectional communication with "first in first out" semantic, hence the name.

You need to create the FIFO with the `int mkfifo(const char *filepath, mode_t mode);` function just like a file, hence it also taking the same mode parameter as when working with files. Opening the FIFO with the `open()` system call in read-only mode or write-only blocks until a second process opens the same FIFO in the other mode. So the two ends of the pipe need to exist. However, you can not open a FIFO with the `O_RDWR` mode.

```c filename="fifo_read.c"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

int main(void)
{
    char *fifoPath = "myfifo";
    if (mkfifo(fifoPath, S_IRUSR | S_IWUSR) == -1 && errno != EEXIST)
    {
        printf("Error when creating FIFO file\n");
        return 1;
    }

    int fd = open(fifoPath, O_RDONLY); // blocks till write end is opened
    int output;
    if (read(fd, &output, sizeof(output)) == -1)
    {
        printf("Error when reading from FIFO");
        return 4;
    }
    printf("PARENT: Received: %d", output);
    remove(fifoPath);
    return 0;
}
```

```c filename="fifo_write.c"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>

int main(void)
{
    char *fifoPath = "myfifo";
    if (mkfifo(fifoPath, S_IRUSR | S_IWUSR) == -1 && errno != EEXIST)
    {
        printf("Error when creating FIFO file\n");
        return 1;
    }
    int fd = open(fifoPath, O_WRONLY); // blocks till other end is opened
    int input = 10;
    if (write(fd, &input, sizeof(input)) == -1)
    {
        printf("Error when writing to FIFO");
        return 3;
    }
    printf("CHILD: Sent: %d", input);
    remove(fifoPath);
    return 0;
}
```

### Non-blocking FIFO

There might be cases where you don't want the open system call to block to avoid deadlocks. To avoid this you can add the `O_NONBLOCK` mode. Opening for read-only will succeed even if the write side hasn't been opened yet. However, opening for write only will return -1 and set `errno=ENXIO` unless the other end has already been opened.

Using `O_NONBLOCK` does have an influence on reading and writing.

If the buffer is empty then the read function returns -1 and sets `errno=EAGAIN`. If additionally the write end is already closed then EOF is returned.

If the read end is not ready yet and the write function is used and fills the buffer then write return -1 and sets `errno=EAGAIN`.
