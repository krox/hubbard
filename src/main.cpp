#include <iostream>
#include <cmath>
#include <cassert>
#include <random>

#include "util/gnuplot.h"

#include "boost/program_options.hpp"
namespace po = boost::program_options;

#include <fmt/format.h>

#include <hubbard.h>

double compute(int n, int l, bool honey, double beta, double U, double mu, int nwarm, int nmeas)
{
	Hubbard hubb;
	if(honey)
		hubb = Hubbard(true, 2*n,n, l);
	else
		hubb = Hubbard(false, n,n, l);

	hubb.setParams(beta, U, mu);
	hubb.clearStats();

	for(int i = 0; i < nwarm; ++i)
	{
		if(nwarm < 20 || i%10 == 0)
			std::cout << "warmup sweep " << i << std::endl;
		hubb.thermalize();
	}

	for(int i = 0; i < nmeas; ++i)
	{
		if(nmeas < 20 || i%10 == 0)
			std::cout << "measurment sweep " << i << std::endl;
		hubb.thermalize();
		hubb.measure();
	}

	std::cout << "n    = " << n << std::endl;
	std::cout << "L    = " << l << std::endl;
	std::cout << "U    = " << U << std::endl;
	std::cout << "beta = " << beta << std::endl;
	hubb.print();

	return hubb.mag();
}

int main(int argc, char** argv)
{
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("n", po::value<int>()->default_value(6), "(spatial) size of lattice")
		("u", po::value<double>()->default_value(4.0), "interaction strength")
		("tMin", po::value<double>()->default_value(0.2), "temperature")
		("tMax", po::value<double>()->default_value(100), "temperature")
		("honey", "honeycomb-lattice")
		("warm", po::value<int>()->default_value(20), "number of warmup sweeps")
		("meas", po::value<int>()->default_value(80), "number of measurment sweeps")
		("trott", po::value<double>()->default_value(0.125), "trotter error")
	;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);
	po::notify(vm);

	if(vm.count("help")){
		std::cout << desc << "\n";
		return 1;
	}

	double mu = 0.0;
	int n = vm["n"].as<int>();
	double trott = vm["trott"].as<double>();

	double U = vm["u"].as<double>();
	double tMin = vm["tMin"].as<double>();
	double tMax = vm["tMax"].as<double>();
	int nwarm = vm["warm"].as<int>();
	int nmeas = vm["meas"].as<int>();

	Gnuplot plot;
	plot.setLabelX("T");
	plot.setLabelY("magetization");
	plot.setLogScaleX();
	plot.setRangeX(tMin, tMax);

	std::vector<double> xs, ys;
	for(double T = tMax; T >= tMin; T *= 0.95)
	{
		// parameters for this run
		double beta = 1/T;
		int l = (int)ceil(beta*sqrt(U/trott));

		// do the simulation
		double mag = compute(n, l, vm.count("honey")!=0, beta, U, mu, nwarm, nmeas);

		// plot the results
		xs.push_back(T);
		ys.push_back(mag);

		if(xs.size() >= 2)
		{
			plot.clear();
			plot.plotData(xs,ys, fmt::format("n={}, U={}", n, U));
			plot.plotFunction([](double t){return 0.5+0.5/t;}, 2.0, tMax);
		}
	}

	plot.writePNG("plot.png");
}
