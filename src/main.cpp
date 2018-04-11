#include <iostream>
#include <cmath>
#include <cassert>
#include <random>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

#include "boost/program_options.hpp"
namespace po = boost::program_options;

#include <hubbard.h>

int main(int argc, char** argv)
{
	po::options_description desc("Allowed options");
	desc.add_options()
		("help", "produce help message")
		("n", po::value<int>()->default_value(6), "(spatial) size of lattice")
		("l", po::value<int>()->default_value(10), "(imaginary time) size of lattice")
		("u", po::value<double>()->default_value(4.0), "interaction strength")
		("beta", po::value<double>()->default_value(1.0), "inverse temperature")
		("honey", po::value<bool>()->default_value(false), "honeycomb-lattice")
		("warm", po::value<int>()->default_value(20), "number of warmup sweeps")
		("meas", po::value<int>()->default_value(80), "number of measurment sweeps")
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
	int l = vm["l"].as<int>();
	double U = vm["u"].as<double>();
	double beta = vm["beta"].as<double>();
	int nwarm = vm["warm"].as<int>();
	int nmeas = vm["meas"].as<int>();

	if(l%blockSize != 0)
	{
		std::cout << "ERROR: L must be a multiple of block-size which is " << blockSize << std::endl;
		return 1;
	}

	Hubbard hubb;
	if(vm["honey"].as<bool>())
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
}
