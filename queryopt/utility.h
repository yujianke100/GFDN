#pragma once
#ifndef __UTILITY_H
#define __UTILITY_H

#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <queue>
#include <map>
#include <string>
#include <algorithm>
#include <boost/functional/hash.hpp>
//#include <stdint.h> //needed to use uint64_t

//typedef unsigned short vid_t;
typedef unsigned int vid_t;
typedef int num_t;

typedef std::unordered_set<std::pair<vid_t,vid_t>, boost::hash<std::pair<vid_t, vid_t>>> batch_set;

/** Macros **/

#define CHAR_BITS  8  /* number of bits in char */

#define XBM  31
#define RSV  5   /* >> 5 == / 32 */

#define IS_SET(vector, bit)   (vector[(bit) >> RSV] & 1 << ((bit) & XBM))
#define SET_BIT(vector, bit)  (vector[(bit) >> RSV] |= 1 << ((bit) & XBM))
#define DEL_BIT(vector, bit)  (vector[(bit) >> RSV] &= ~(1 << ((bit) & XBM)))

#define SET_VALUE(vector, index, value)  (vector[index] = value)
#define GET_VALUE(vector, index)         (vector[index])

#define MIN(a, b) (a <= b ? a : b)
#define MAX(a, b) (a >= b ? a : b)

#define MAX_NUM_CHUNK 4

#define PRINT_INDEX_DETAIL false

/* Macros for bit operations */

/* return the number of 1's in a given bit string */
extern unsigned char bit_num_ones[256];
#define bit_count_ones(n, b, s) {\
  unsigned char *bits = (unsigned char *)b;\
  int my_i;\
  n = 0; \
  for (my_i = 0; my_i < s; my_i++) \
    n += bit_num_ones[bits[my_i]];\
}

/* return 1 if all bits are 0's, return 0 otherwise */
#define bit_all_zeros(b, s, yn) {\
  unsigned int *bits;\
  int my_i;\
  bits = (unsigned int *)b;\
  for (my_i = 0; my_i < s/sizeof(unsigned int); my_i++) \
    if (bits[my_i] != 0) { yn = 0; break; } \
}

/* return the intersection of two memory block of given size to dst */
#define bit_intersect(dst, src, n) {\
  int my_i, my_n=n/sizeof(unsigned int); \
  if (dst != NULL && src != NULL) {\
	for (my_i = 0; my_i < my_n; my_i++) \
      dst[my_i] &= src[my_i]; }\
}

/* return the union of two memory block of given size to dst */
#define bit_union(dst, src, n) {\
  int my_i, my_n=n/sizeof(unsigned int); \
  for (my_i = 0; my_i < my_n; my_i++) \
    dst[my_i] != src[my_i]; \
}

/* return the number of unsigned ints to represent n bits */
#define bit_num_ints(n) (n / (sizeof(unsigned int)*CHAR_BITS) + \
(n % (sizeof(unsigned int)*CHAR_BITS) == 0 ? 0 : 1))

/* return the number of chars to represent n bits */
#define bit_num_chars(n) (sizeof(unsigned int) * bit_num_ints(n))



/** Timer function **/

/* Returns current time in seconds as a double value
and the precision can be as accurate as microseconds
(10^-6 second)
*/
double get_cur_time();

enum class Index_update {withlimit, withlimit_base_opt, withlimit_parallel, withlimit_dfs, withlimit_dfs_parallel, withoutlimit, batch, batch_parallel};

extern std::ostream& operator<<(std::ostream& out, const Index_update value);

enum class Datasets { BX, TM, AM, WC, FG,
	PA, EP, DE, RE, TR,
	DUI, LJ, WT, OG
};

extern std::ostream& operator<<(std::ostream& out, const Datasets value);

extern std::map<Datasets, std::string> dataset2name;

#endif  /* __UTILITY_H */