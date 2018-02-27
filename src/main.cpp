#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <cassert>
#include <random>
#include <unsupported/Eigen/MatrixFunctions>
using Eigen::MatrixXd;
using Eigen::VectorXd;


class Hubbard
{
	// random number generator
	std::random_device rd;
	std::mt19937 rng;

public:

	int N,L; // geometry of the lattice
	double beta,U,mu; // parameters of Hamiltonian
	double dt,lambda; // effective parameters of aux field

	MatrixXd K, expK; // kinetic operator (fixed during simulation)
	MatrixXd s; // aux field (dynamic variable)
	MatrixXd gu,gd; // greens functions (computed from s and K)

	Hubbard(int Nx, int Ny, int L, double beta, double U, double mu)
		: rng(rd()), N(Nx*Ny), L(L), K(N,N), expK(N,N), s(N,L), gu(N,N), gd(N,N)
	{
		// set off-diagonal elements of K
		// which only depend on geomety, not on parameters (in our convention)
		for(int i = 0; i < N; ++i)
			for(int j = 0; j < N; ++j)
				K(i,j) = 0.0;
		for(int x = 0; x < Nx; ++x)
			for(int y = 0; y < Ny; ++y)
			{
				K(x + Nx*y, ((x+1)%Nx) + Nx*y) = 1.0;
				K(((x+1)%Nx) + Nx*y,x + Nx*y) = 1.0;
				K(x + Nx*y, x + Nx*((y+1)%Ny)) = 1.0;
				K(x + Nx*((y+1)%Ny), x + Nx*y) = 1.0;
			}

		initRandom();
		setParams(beta, U, mu);
	}

	/** set simulation parameters without changing current field config */
	void setParams(double beta, double U, double mu)
	{
		// TODO: actually support chemical potential (requires reweighting)
		assert(mu == 0);

		// set parameters
		this->beta = beta;
		this->U = U;
		this->mu = mu;
		this->dt = beta/L;
		this->lambda = acosh(exp(0.5*dt*U));

		// compute new kinetic operator
		// (note: the off-diagonal elements are fixed by geometry)
		for(int i = 0; i < N; ++i)
			K(i,i) = mu;
		expK = (-dt*K).exp();

		// greens functions depend on K, so update them
		computeGreens();
	}

	/** set the field config to random */
	void initRandom()
	{
		std::bernoulli_distribution dist(0.5);
		for(int i = 0; i < N; ++i)
			for(int l = 0; l < L; ++l)
				s(i,l) = dist(rng)?+1.0:-1.0;
	}

	/** compute greens functions from scratch */
	void computeGreens()
	{
		gu = MatrixXd::Identity(N,N);
		gd = MatrixXd::Identity(N,N);
		for(int l = 0; l < L; ++l)
		{
			gu = makeBl(l,+1)*gu;
			gd = makeBl(l,-1)*gd;
		}
		gu = (MatrixXd::Identity(N,N)+gu).inverse();
		gd = (MatrixXd::Identity(N,N)+gd).inverse();
	}

	MatrixXd makeBl(int l, int sigma)
	{
		MatrixXd r = expK;
		for(int i = 0; i < N; ++i)
			r.col(i) *= exp(sigma*lambda*s(i,l));
		return r;
	}

	/** do one sweep of the simulation */
	void thermalize()
	{
		// start with fresh greens functions
		computeGreens();

		for(int l = 0; l < L; ++l)
		{
			for(int i = 0; i < N; ++i)
			{
				// propose to flip s[i,l]
				double p = (1 + (1-gu(i,i))*(exp(-2*lambda*s(i,l))-1))
				         * (1 + (1-gd(i,i))*(exp(+2*lambda*s(i,l))-1));
				std::bernoulli_distribution dist(std::min(1.0,p));
				if(dist(rng))
				{
					// update greens function (which is wrapped such that the update is at timeslice 0 of g)
					double factorU = (exp(-2*lambda*s(i,l))-1)/(1 + (1-gu(i,i))*(exp(-2*lambda*s(i,l))-1));
					gu += factorU*(gu.col(i)-VectorXd::Unit(N,i))*gu.row(i);

					double factorD = (exp(+2*lambda*s(i,l))-1)/(1 + (1-gd(i,i))*(exp(+2*lambda*s(i,l))-1));
					gd += factorD*(gd.col(i)-VectorXd::Unit(N,i))*gd.row(i);

					// update field itself
					s(i,l) = -s(i,l);
				}
			}

			// 'wrap' the greens functions
			MatrixXd Blu = makeBl(l,+1);
			gu = Blu * gu * Blu.inverse();
			MatrixXd Bld = makeBl(l,-1);
			gd = Bld * gd * Bld.inverse();
		}

		// end with fresh greens functions
		computeGreens();
	}

	double nn = 0.0;
	double nu = 0.0;
	double nd = 0.0;
	double nud = 0.0;

	void measure()
	{
		for(int i = 0; i < N; ++i)
		{
			nn += 1.0;
			nu += 1.0-gu(i,i);
			nd += 1.0-gd(i,i);
			nud += (1.0-gu(i,i))*(1.0-gd(i,i));
		}
	}

	void print()
	{
		std::cout << "<up> = " << nu/nn << std::endl;
		std::cout << "<down> = " << nu/nn << std::endl;
		std::cout << "<up*down> = " << nud/nn << std::endl;
	}
};

int main()
{
	auto hubb = Hubbard(6,6,10,1.0,4.0,0.0);
	for(int i = 0; i < 200; ++i)
	{
		std::cout << "thermlize " << i << std::endl;
		hubb.thermalize();
	}
	for(int i = 0; i < 800; ++i)
	{
		std::cout << "thermlize+measure " << i << std::endl;
		hubb.measure();
		hubb.thermalize();
	}

	hubb.print();
}
