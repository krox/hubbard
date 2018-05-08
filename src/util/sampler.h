#ifndef MESH_SAMPLER_H
#define MESH_SAMPLER_H

#include <cmath>
#include <queue>
#include <vector>

#include "boost/math/tools/roots.hpp"
using boost::math::tools::eps_tolerance;
using boost::math::tools::toms748_solve;

#ifdef TEST_SAMPLER
#include "util/gnuplot.h"
#include "util/random.h"
#include <iostream>
#endif

struct LogSampler_Region
{
	double a, b; // left/right bounds of region

	// upper/lower bound of f in the interval [a,b]
	// alpha + beta*x >= f(a+x(b-a)) >= gamma + delta*x
	double alpha, beta;
	double gamma, delta;

	/** area of upper bound */
	double areaUpper() const
	{
		if (beta == 0)
			return (b - a) * exp(alpha);
		return (b - a) / beta * exp(alpha) * expm1(beta);
	}

	/** area of lower bound */
	double areaLower() const
	{
		if (delta == 0)
			return (b - a) * exp(gamma);
		return (b - a) / delta * exp(gamma) * expm1(delta);
	}

	/** lower area / upper area */
	double frac() const { return areaLower() / areaUpper(); }

	/** sort regions with bad bounds to the front */
	bool operator<(const LogSampler_Region &r) const
	{
		return areaUpper() - areaLower() < r.areaUpper() - r.areaLower();
	}
};

/**
 * Sampler for arbitrary distribution
 */
template <typename F, typename FD, typename FDD> class LogSampler
{
	F f;
	FD fd;
	FDD fdd;

	std::vector<LogSampler_Region> regs;

	std::discrete_distribution<int> disc_dist;
	std::uniform_real_distribution<double> uni_dist;

	size_t nTries = 0;
	size_t nEvals = 0;
	size_t nAccepts = 0;

	LogSampler_Region makeRegion(double a, double b)
	{
		assert(a < b);
		LogSampler_Region r;
		r.a = a;
		r.b = b;

		// secant through endpoints
		r.alpha = f(a);
		r.beta = f(b) - f(a);
		if (r.alpha == -std::numeric_limits<double>::infinity())
			r.beta = 0;

		// tangent to midpoint
		r.delta = fd(0.5 * (a + b)) * (b - a);
		r.gamma = f(0.5 * (a + b)) - 0.5 * r.delta;

		if (r.alpha < r.gamma)
		{
			std::swap(r.alpha, r.gamma);
			std::swap(r.beta, r.delta);
		}

		assert(r.alpha >= r.gamma);
		assert(r.alpha + r.beta >= r.gamma + r.delta);
		return r;
	}

  public:
	double min, max;

	LogSampler(F f, FD fd, FDD fdd, double min, double max, size_t nRegs = 50)
	    : f(f), fd(fd), fdd(fdd), uni_dist(0, 1), min(min), max(max)
	{
		assert(min < max);

		// sample f'' at 100 points
		std::vector<double> xs;
		xs.push_back(min);
		for (int i = 1; i < 99; ++i)
			xs.push_back(min + (max - min) * i / 99);
		xs.push_back(max);
		std::vector<double> fddxs;
		for (size_t i = 0; i < xs.size(); ++i)
			fddxs.push_back(fdd(xs[i]));

		// find inflection points to use as initial region bounds
		boost::uintmax_t max_iter = 100;
		std::vector<double> ys;
		ys.push_back(xs.front());
		for (size_t i = 1; i < xs.size(); ++i)
			if (fddxs[i - 1] * fddxs[i] < 0)
			{
				auto tol = eps_tolerance<double>(100);
				boost::uintmax_t cnt = 100;
				double y = toms748_solve(fdd, xs[i - 1], xs[i], tol, cnt).first;
				ys.push_back(y);
			}

		// create initial regions
		std::priority_queue<LogSampler_Region> heap;
		ys.push_back(xs.back());
		for (size_t i = 1; i < ys.size(); ++i)
			heap.push(makeRegion(ys[i - 1], ys[i]));

		// split regions that have bad bounds
		while (heap.size() < nRegs)
		{
			auto r = heap.top();
			heap.pop();
			double m = 0.5 * (r.a + r.b);
			heap.push(makeRegion(r.a, m));
			heap.push(makeRegion(m, r.b));
		}

		while (!heap.empty())
		{
			regs.push_back(heap.top());
			heap.pop();
		}

		// sort regions (not actually neccessary, but looks nicer for debugging)
		std::sort(regs.begin(), regs.end(),
		          [](const LogSampler_Region &a, const LogSampler_Region &b) {
			          return a.a < b.a;
		          });

		// create distribution of regions
		std::vector<double> areas;
		for (auto &r : regs)
			areas.push_back(r.areaUpper());
		disc_dist = std::discrete_distribution<int>(areas.begin(), areas.end());
	}

	template <typename Rng> double operator()(Rng &rng)
	{
		while (true)
		{
			++nTries;

			// choose a region
			size_t i = disc_dist(rng);
			const LogSampler_Region &r = regs[i];

			// candidate in [0,1] with exponential distribution
			double u = uni_dist(rng) * (1 - exp(-r.beta)) + exp(-r.beta);
			double x = log(u) / r.beta + 1;
			if (r.beta == 0)
				x = uni_dist(rng);
			assert(0 <= x && x <= 1);

			// uniform in [0,upper(x)]
			double y = uni_dist(rng) * exp(r.alpha + r.beta * x);

			if (y > exp(r.gamma + r.delta * x)) // over lower bound?
			{
				++nEvals;
				if (y > exp(f(r.a + (r.b - r.a) * x))) // over f itself?
					continue;
			}

			++nAccepts;
			return regs[i].a + (r.b - r.a) * x;
		}
	}

	double accProb() const { return (double)nAccepts / nTries; }
	double evalProb() const { return (double)nEvals / nTries; }

#ifdef TEST_SAMPLER

	void test()
	{
		{
			xoroshiro128plus rng{std::random_device()()};

			// create a histogram
			histogram hist(min, max, 50);
			for (int i = 0; i < 10000000; ++i)
				hist.add((*this)(rng));

			// match scaling at largest bin
			int maxBin = 0;
			for (size_t i = 1; i < hist.bins.size(); ++i)
				if (hist.bins[i] > hist.bins[maxBin])
					maxBin = i;

			double maxX = 0.5 * (hist.mins[maxBin] + hist.maxs[maxBin]);
			double c = hist.bins[maxBin] / exp(f(maxX));

			Gnuplot p;
			p.plotHistogram(hist);
			p.plotFunction([&](double x) { return c * exp(f(x)); }, min, max);
		}

		{
			std::vector<double> plotX, plotY, plotY2;
			for (auto &r : regs)
			{
				plotX.push_back(r.a);
				plotX.push_back(r.b);
				plotY.push_back(r.gamma);
				plotY.push_back(r.gamma + r.delta);
				plotY2.push_back(r.alpha);
				plotY2.push_back(r.alpha + r.beta);
			}

			Gnuplot plot;
			plot.style = "lines";
			plot.plotFunction(f, min, max, "log-prob");
			plot.plotData(plotX, plotY, "min");
			plot.plotData(plotX, plotY2, "max");
		}

		std::cout << "nRegs    = " << regs.size() << std::endl;
		std::cout << "accProb  = " << accProb() << std::endl;
		std::cout << "evalProb = " << evalProb() << std::endl;
	}

#endif
};

#endif
