---
title: POSIX Interprocess Communication
tags: [c, ipc, processes, pipes, posix, synchronization, semaphores]
---

# POSIX Interprocess Communication

## POSIX Semaphores

You can find a detailed explanation of what a semaphore is [here](../../Concurrent%20Programming/9-synchronizers.md).

### Named Semaphores

Named semaphores have a name and can be used by multiples process just like named pipes (FIFOs). The process that opens the semaphore but doesn't create it just needs to pass the first 2 arguments. Just like with mutexes there are in addition the functions `sem_trywait(sem_t *sem)` and `sem_timedwait(sem_t *sem, const struct timespec *abs_timeout);`.

```c
#include <fcntl.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <stdio.h>

int main(void)
{
    char *name = "/my_semaphore";                                // must start with "/""
    sem_t *sema = sem_open(name, O_CREAT, S_IRUSR | S_IRGRP, 2); // or/and O_EXCL
    sem_wait(sema);
    // sem_wait(sema); // blocks
    int current = 0;
    sem_getvalue(sema, &current);
    printf("Decrease semaphore by 1, now: %d\n", current);
    sem_post(sema);
    sem_getvalue(sema, &current);
    printf("Add semaphore by 1, now: %d\n", current);
    sem_close(sema);
    sem_unlink(name);
    return 0;
}
```

### Unnamed Semaphores

Unnamed semaphores work the same way as named ones but they are in memory and can be accessed by processes and threads via shared memory. Instead of opening one you need to initialize it with the `int sem_init(sem_t *sem, int pshared, unsigned int value);` function and when you are finished with it remove it with `int sem_destroy(sem_t *sem);`. The pshared argument indicates whether this semaphore is to be shared between the threads of a process, or between processes. If pshared has the value 0, then the semaphore is shared between the threads of a process, and should be located at some address that is visible to all threads. If pshared is nonzero, then the semaphore is shared between processes, and should be located in POSIX shared memory.

## POSIX Shared Memory

With POSIX shared memory you can have shared memory between processes without using files.

/// TODOOOOOO an example that actually works///

```c
#include <stdio.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

int main(void)
{
    char* name = "/my_shm1";
    int data = 10;

    int fd = shm_open(name, O_CREAT | O_RDWR, S_IRUSR|S_IRGRP);
    if (fd == -1)
    {
        printf("Failed to create shm object");
        return 1;
    }
    ftruncate(data, sizeof(int));
    // map shared memory to process address space
    void *addr = mmap(NULL, sizeof(int), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED)
    {
        printf("Failed to map shm object");
        return 2;
    }

    int id = fork();
    if (id == -1)
    {
        printf("Failed to fork");
        return 3;
    }
    if (id == 0) // child process
    {
        data = 15;
    }
    else // parent process
    {
        wait(NULL); // wait for update to take effect
    }
    printf("data is: %d\n", data);
    shm_unlink(name);
    return 0;
}
```
