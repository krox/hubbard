#ifndef MESH_RANDOM_H
#define MESH_RANDOM_H

/**
 * Random number generators to be used in the Monte-Carlo simulation.
 */

#include <stdint.h>

/**
 * Originally written in 2015 by Sebastiano Vigna (vigna@acm.org).
 * public domain, taken from http://xoroshiro.di.unimi.it/splitmix64.c
 */
class splitmix64
{
	uint64_t s; // all values are allowed

  public:
	splitmix64() : s(0) {}
	explicit splitmix64(uint64_t x) : s(x) {}

	uint64_t min() const { return 0; }
	uint64_t max() const { return UINT64_MAX; }

	void seed(uint64_t x) { s = x; }

	uint64_t operator()()
	{
		s += 0x9e3779b97f4a7c15;
		uint64_t z = s;
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		return z ^ (z >> 31);
	}
};

/**
 * Originally written in 2016 by David Blackman and Sebastiano Vigna
 * public domain, taken from http://xoroshiro.di.unimi.it/xoroshiro128plus.c
 */
class xoroshiro128plus
{
	uint64_t s[2]; // should not be all zeroes

	static inline uint64_t rotl(uint64_t x, int k)
	{
		// NOTE: compiler will optimize this to a single instruction
		return (x << k) | (x >> (64 - k));
	}

  public:
	xoroshiro128plus() { seed(0); }
	explicit xoroshiro128plus(uint64_t x) { seed(x); }

	uint64_t min() const { return 0; }
	uint64_t max() const { return UINT64_MAX; }

	/** set the internal state to some seed value */
	void seed(uint64_t x)
	{
		splitmix64 gen(x);
		s[0] = gen();
		s[1] = gen();
	}

	/** generate next value in the random sequence */
	uint64_t operator()()
	{
		uint64_t s0 = s[0];
		uint64_t s1 = s[1];
		uint64_t result = s0 + s1;

		s1 ^= s0;
		s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
		s[1] = rotl(s1, 36);

		return result;
	}

	/** discards 2^64 values of the random sequence */
	void jump()
	{
		static const uint64_t JUMP[] = {0xbeac0467eba5facb, 0xd86b048b86aa9922};

		uint64_t s0 = 0;
		uint64_t s1 = 0;
		for (int i = 0; i < 2; i++)
			for (int b = 0; b < 64; b++)
			{
				if (JUMP[i] & UINT64_C(1) << b)
				{
					s0 ^= s[0];
					s1 ^= s[1];
				}
				(*this)();
			}

		s[0] = s0;
		s[1] = s1;
	}
};

#endif
