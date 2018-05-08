#include "util/stats.h"

#include <algorithm>

histogram::histogram(double min, double max, size_t n)
    : mins(n), maxs(n), bins(n, 0)
{
	for (size_t i = 0; i < n; ++i)
	{
		mins[i] = min + (max - min) * i / n;
		maxs[i] = min + (max - min) * (i + 1) / n;
	}
}

void histogram::add(double x)
{
	auto it = std::upper_bound(maxs.begin(), maxs.end(), x);
	auto i = std::distance(maxs.begin(), it);
	if (0 <= i && i < (ptrdiff_t)bins.size()) // ignore out-of-range samples
		bins[i] += 1;
}
