---
title: Threads and Synchronization with C
tags: [c, threads, posix, pthread, synchronization, mutex, locks, conditional variables]
---

# Threads and Synchronization with C

Threads are similar to process and allow a program to do multiple things at once by having multiple threads in it. A key difference between threads and process is however that threads share the same global memory and just have their private stack for local variables and function calls and are therefore not as expensive as process which have the big overhead of creating an entire new memory space. This is why threads are also often called lightweight processes.

Exchanging information between process can be quiet tricky and costly because the parent and child don't share memory. However, threads share the following things between each other amongst other things

- PID and Parent PID
- Open file descriptors
- Signal handlers
- Global memory

Each thread however does receive the following things for itself

- Thread ID
- Signal Mask
- Errno variable
- Stack

To work with threads you can use the Pthreads API which are also known as POSIX threads, sounds familiar... which is provided with the gcc compiler. Important to know here is that Pthreads functions don't return -1 on failure like many other functions in the standard library. Instead they return 0 on success and add the errno on failure. To be able to use the Pthreads API you need to pass the `-pthread` flag to the gcc compiler.

## Creating Threads

To create a thread you will need to use the following function

```c
int pthread_create(pthread_t *restrict thread,
                        const pthread_attr_t *restrict attr,
                        void *(*start_routine)(void *),
                        void *restrict arg);
 ```

The first parameter is an integer that is used as an output parameter and is used to identify the thread in your operating system.
The second parameter is for specific attributes for the thread, by passing NULL you can use the default.
The third parameter is the function that the thread will execute once it is started.
The fourth parameter is used to pass arguments to the function and must be cast to a void pointer. If you want to pass multiple arguments, you would use a pointer to a struct.

```c
int pthread_join(pthread_t thread, void **retval);
```

A call to the join function blocks the calling thread until the thread with ID as the first parameter is terminated. You can also store the return value of the thread with the second parameter.

Threads can terminate in multiple ways

- By calling `void pthread_exit(void *retval);`
- By letting the thread function return.
- By calling exit which will terminate the process including all its threads.

Interestingly of the main thread calls pthread_exit all the other threads will continue to execute otherwise they all automatically terminate when main returns.

```c
#include <stdio.h>
#include <pthread.h>

#include <stdio.h>
#include <pthread.h>

void *foo()
{
    printf("foo ID: %ld\n", pthread_self());
    pthread_exit(NULL);
}

int main(void)
{
    printf("main ID: %ld\n", pthread_self());
    pthread_t foo_t;
    pthread_create(&foo_t, NULL, foo, NULL);

    pthread_join(foo_t, NULL);
    printf("done");

    return 0;
}

```

## Passing Values

When creating the pthread you can pass the arguments using the fourth parameter.

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct Point
{
    double x;
    double y;
} Point;

void *printPoint(void *args)
{
    Point p = *((Point *)args);
    printf("(%f, %f)", p.x, p.y);
    pthread_exit(NULL);
}
int main(void)
{
    pthread_t pid;
    Point p = {2, 10};

    pthread_create(&pid, NULL, printPoint, &p);
    pthread_join(pid, NULL);
    return 0;
}
```

## Returning Values

You can return values from a thread with the pthread_exit function. The values you return should be on the heap otherwise you will run into problems.

```c
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct Point
{
    double x;
    double y;
} Point;

void *createPoint()
{
    Point *p = malloc(sizeof(Point));
    p->x = 3;
    p->y = 7;
    pthread_exit((void *)p);
}
int main(void)
{
    pthread_t pid;
    Point p;
    void *res;
    pthread_create(&pid, NULL, createPoint, NULL);
    pthread_join(pid, &res);

    p = *((Point *)res);
    free(res);
    res = NULL;
    printf("(%f, %f)", p.x, p.y);
    return 0;
}
```

## Further Operations on Threads

There are further operations that you can use with threads for example:

- `int pthread_equal(pthread_t t1, pthread_t t2);` which compares two threads to see whether they are the same.
- `int pthread_detach(pthread_t thread);` by default a thread runs in joinable mode. A joinable thread will not release its resources even after termination until some other thread calls `pthread_join()` with its ID. A Detached thread automatically releases its allocated resources on exit. No other thread needs to join it. Therefore there is also no way to determine its return value.
- `int pthread_cancel(pthread_t thread);` sends a cancellation request to the thread. Whether the target thread reacts to the cancellation request depends on its cancelability state and type.
- `int pthread_kill(pthread_t thread, int sig);` sends the signal sig to thread.

## Synchronization

### Mutex

Threads can have a mutual state which is useful but you need to be careful when accessing and changing this state. A critical section is a code block that uses a mutual variable and should only be executed atomically, so at once, by one thread, so that the result does not depend on the interleaving. A mutex/lock can guarantee this behavior to avoid race conditions. For more on this, there is an entire [section dedicated to concurrent programming](../../concurrentProgramming/locking).

Mutex variables are of the type `pthread_mutex_t` and need to be initialized before they are used with `pthrad_mutex_t m = PTHREAD_MUTEX_INITIALIZER;` or the `int pthread_mutex_init(pthread_mutex_t *restrict mutex, const pthread_mutexattr_t *restrict attr);` function. To then use the lock you can use the following functions:

- `int pthread_mutex_lock(pthread_mutex_t *mutex);` Acquires the lock. If the lock is already in use then this function blocks till it can acquire the lock. If called repeatedly in the same thread that already has the lock a deadlock occurs.
- `int pthread_mutex_unlock(pthread_mutex_t *mutex);` Releases the lock. If called in a thread that has already released a lock will return an error.
- `int pthread_mutex_trylock(pthread_mutex_t *mutex);` Tries to acquire the lock. If it can't it does not block. Instead, it returns `EBUSY`.
- `int pthread_mutex_timedlock(pthread_mutex_t *restrict mutex, const struct timespec *restrict abstime);` Tries to acquire the lock and waits for a maximum of abstime. If it couldn't get acquire the lock in the given time it returns `ETIMEDOUT`.

With the timespec struct looking like this:

```c
struct timespec {
    time_t tv_sec; // seconds
    long tv_nsec; // nanoseconds
};
```

An example of using locks:

```c
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

pthread_t tid[2];
int counter;
pthread_mutex_t lock; // lock object

void *incrementCounter()
{
    for (unsigned long i = 0; i < 1000; i++)
    {
        pthread_mutex_lock(&lock);
        counter++;
        pthread_mutex_unlock(&lock);
    }
    pthread_exit(NULL);
}

int main(void)
{
    if (pthread_mutex_init(&lock, NULL) != 0) // init lock
    {
        printf("\n mutex init failed\n");
        return 1;
    }
    int i = 0;
    while (i < 2)
    {
        int err = pthread_create(&(tid[i]), NULL, &incrementCounter, NULL);
        if (err != 0)
            printf("\ncan't create thread :[%s]", strerror(err));
        i++;
    }
    // wait for threads to finish
    pthread_join(tid[0], NULL);
    pthread_join(tid[1], NULL);
    printf("Counter: %d", counter);
    pthread_mutex_destroy(&lock); // clean up lock

    return 0;
}
```

### Condition Variables

In C we can also use [condition variables](../../concurrentProgramming/conditionVariables) to further synchronize concurrent programs. Condition variables in C work just like in Java. `pthread_cond_signal` is the equivalant of `notify()` and `pthread_cond_broadcast` of `notifyAll()`.

```c
pthread_mutex_t lock; // init with PTHREAD_MUTEX_INITIALIZER
pthread_cond_t count_nonzero; // init with PTHREAD_COND_INITIALIZER
unsigned int count;

decrementCount()
{
    pthread_mutex_lock(&lock);
    while (count == 0)
        pthread_cond_wait(&count_nonzero, &lock);
    count = count - 1;
    pthread_mutex_unlock(&lock);
}

incrementCount(){
    pthread_mutex_lock(&lock);
    count++;
    if(count == 0){
        pthread_cond_broadcast(&count_nonzero);
        // pthread_cond_signal(&count_nonzero)
    }
    pthread_mutex_unlock(&lock);
}
````

## Errno

To be able to set errno in a thread-safe manner we need to use a makro from the pthread library

```c
# define errno (*__errno_location())
*__errno_location() = EBUSY;
```
