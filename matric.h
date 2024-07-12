// matric.h

#ifndef MATRIC_H
#define MATRIC_H

// Macro definitions
#define THROUGHPUT(operations, seconds) ((operations) / (seconds) / 1e9) // GOPS
#define RATIO_TO_PEAK_BANDWIDTH(actual_bandwidth, peak_bandwidth) ((actual_bandwidth) / (peak_bandwidth))

#endif // MATRIC_H

