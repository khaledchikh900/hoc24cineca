#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

static struct timeval timerStart;

void StartTimer() {
    gettimeofday(&timerStart, NULL);
}

double GetTimer() {
    struct timeval timerEnd;
    gettimeofday(&timerEnd, NULL);
    return (double)(timerEnd.tv_sec - timerStart.tv_sec) * 1000.0 + (double)(timerEnd.tv_usec - timerStart.tv_usec) / 1000.0;
}

#endif // TIMER_H

