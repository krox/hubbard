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
public:
	std::random_device rd;
	std::mt19937 rng;

	int Nx,Ny,N,L;
	double beta,U,mu,dt,lambda;

	MatrixXd K, expK;
	MatrixXd s, expVu, expVd;
	MatrixXd gu,gd;

	Hubbard(int Nx, int Ny, int L)
		: rng(rd()), Nx(Nx), Ny(Ny), N(Nx*Ny), L(L), K(N,N), expK(N,N), s(N,L), expVu(N,L), expVd(N,L), gu(N,N), gd(N,N)
	{
		// input parameters
		beta = 1.0;
		U = 4.0;
		mu = 0.0;

		// constants
		N = Nx*Ny;
		dt = beta/L;
		lambda = acosh(exp(dt*U/2));

		// kinetic matrix K (independant of l)
		for(int i = 0; i < N; ++i)
			for(int j = 0; j < N; ++j)
				K(i,j) = 0.0;
		for(int x = 0; x < Nx; ++x)
			for(int y = 0; y < Ny; ++y)
			{
				K(x + Nx*y, x + Nx*y) = mu;
				K(x + Nx*y, ((x+1)%Nx) + Nx*y) = 1.0;
				K(((x+1)%Nx) + Nx*y,x + Nx*y) = 1.0;
				K(x + Nx*y, x + Nx*((y+1)%Ny)) = 1.0;
				K(x + Nx*((y+1)%Ny), x + Nx*y) = 1.0;
			}
		expK = (-dt*K).exp();//matExp(-dt*K);

		// aux field s (random init)
		std::bernoulli_distribution dist(0.5);
		for(int i = 0; i < N; ++i)
			for(int l = 0; l < L; ++l)
				s(i,l) = dist(rng)?+1.0:-1.0;

		// potential 'matrix' (diagonal in i)
		for(int i = 0; i < N; ++i)
			for(int l = 0; l < L; ++l)
			{
				expVu(i,l) = exp(+lambda*s(i,l));
				expVd(i,l) = exp(-lambda*s(i,l));
			}

		computeGreens();
	}

	void computeGreens()
	{
		gu = MatrixXd::Identity(N,N);
		gd = MatrixXd::Identity(N,N);
		for(int l = 0; l < L; ++l)
		{
			gu = expK * expVu.col(l).asDiagonal() * gu;
			gd = expK * expVd.col(l).asDiagonal() * gd;
		}
		gu = (MatrixXd::Identity(N,N)+gu).inverse();
		gd = (MatrixXd::Identity(N,N)+gd).inverse();
	}

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
					expVu(i,l) = exp(+lambda*s(i,l));
					expVd(i,l) = exp(-lambda*s(i,l));
				}
			}

			// 'wrap' the greens functions
			MatrixXd Blu = expK * expVu.col(l).asDiagonal();
			gu = Blu * gu * Blu.inverse();
			MatrixXd Bld = expK * expVd.col(l).asDiagonal();
			gd = Bld * gd * Bld.inverse();
		}
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
	auto hubb = Hubbard(6,6,10);
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
