#include <iostream>
#include <cmath>
#include <cassert>
#include <random>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
using Eigen::MatrixXd;
using Eigen::VectorXd;

class Hubbard
{
	// random number generator
	static std::random_device rd;
	std::mt19937 rng;

public:
	int Nx,Ny;
	int N,L; // geometry of the lattice
	double beta,U,mu; // parameters of Hamiltonian
	double dt,lambda; // effective parameters of aux field

	MatrixXd K, expK; // kinetic operator (fixed during simulation)
	MatrixXd s; // aux field (dynamic variable)
	MatrixXd gu,gd; // greens functions (computed from s and K)

	Hubbard() = default;
	Hubbard(bool honeycomb, int Nx, int Ny, int L);

	/** set simulation parameters without changing current field config */
	void setParams(double beta, double U, double mu);

	/** set the field config to random */
	void initRandom();

	/** compute greens functions from scratch */
	void computeGreensNaive(int l0, int sigma);

	/** ditto, with SVD decomposition */
	void computeGreens(int l0, int sigma);

	MatrixXd makeBl(int l, int sigma);

	/** do one sweep of the simulation */
	void thermalize();

	double nn = 0.0;
	double nu = 0.0;
	double nd = 0.0;
	double nud = 0.0;

	void measure();

	void clearStats();

	void print() const;

	double mag() const;
};
