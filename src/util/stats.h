#ifndef MESH_STATS_H
#define MESH_STATS_H

#include <vector>

/** statistics utilities */

struct histogram
{
	std::vector<double> mins;
	std::vector<double> maxs;
	std::vector<size_t> bins;

	histogram(double min, double max, size_t n);
	histogram(const std::vector<double> &xs);

	void add(double x);
};

#endif
