#include <hubbard.h>

std::random_device Hubbard::rd;

Hubbard::Hubbard(bool honeycomb, int Nx, int Ny, int L)
	: rng(rd()), Nx(Nx), Ny(Ny), N(Nx*Ny), L(L), K(N,N), expK(N,N), s(N,L), gu(N,N), gd(N,N)
{
	// set off-diagonal elements of K
	// which only depend on geomety, not on parameters (in our convention)
	for(int i = 0; i < N; ++i)
		for(int j = 0; j < N; ++j)
			K(i,j) = 0.0;
	if(honeycomb)
	{
		assert(Nx == 2*Ny);
		for(int x = 0; x < Nx; ++x)
			for(int y = 0; y < Ny; ++y)
			{
				K(x + Nx*y, ((x+1)%Nx) + Nx*y) = 1.0;
				K(((x+1)%Nx) + Nx*y, x + Nx*y) = 1.0;
				if(x%2 == 0)
				{
					K(x + Nx*y, x + Nx*((y+1)%Ny)) = 1.0;
					K(x + Nx*((y+1)%Ny), x + Nx*y) = 1.0;
				}
			}
	}
	else
	{
		assert(Nx == Ny);
		for(int x = 0; x < Nx; ++x)
			for(int y = 0; y < Ny; ++y)
			{
				K(x + Nx*y, ((x+1)%Nx) + Nx*y) = 1.0;
				K(((x+1)%Nx) + Nx*y, x + Nx*y) = 1.0;
				K(x + Nx*y, x + Nx*((y+1)%Ny)) = 1.0;
				K(x + Nx*((y+1)%Ny), x + Nx*y) = 1.0;
			}
	}
	initRandom();
}

/** set simulation parameters without changing current field config */
void Hubbard::setParams(double beta, double U, double mu)
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
	computeGreens(0, +1);
	computeGreens(0, -1);
}

/** set the field config to random */
void Hubbard::initRandom()
{
	std::bernoulli_distribution dist(0.5);
	for(int i = 0; i < N; ++i)
		for(int l = 0; l < L; ++l)
			s(i,l) = dist(rng)?+1.0:-1.0;
}

/** compute greens functions from scratch */
void Hubbard::computeGreensNaive(int l0, int sigma)
{
	MatrixXd g = MatrixXd::Identity(N,N);
	for(int l = 0; l < L; ++l)
		g = makeBl((l+l0)%L, sigma)*g;

	g = (MatrixXd::Identity(N,N) + g).inverse();

	if(sigma == +1) gu = g;
	else if(sigma == -1) gd = g;
	else assert(false);
}

/** ditto, with QR decomposition */
void Hubbard::computeGreens(int l0, int sigma)
{
	// SVD decomposition B = USV*
	MatrixXd B0 = makeBlockBl(l0%L, sigma);
	auto svd = B0.bdcSvd(Eigen::ComputeFullU|Eigen::ComputeFullV);
	VectorXd S = svd.singularValues();
	MatrixXd U = svd.matrixU();
	MatrixXd V = svd.matrixV();

	for(int l = blockSize; l < L; l += blockSize)
	{
		MatrixXd Bl = makeBlockBl((l+l0)%L, sigma);
		MatrixXd C = (Bl*U)*S.asDiagonal();
		svd = C.bdcSvd(Eigen::ComputeFullU|Eigen::ComputeFullV);
		S = svd.singularValues();
		U = svd.matrixU();
		V = V*svd.matrixV();
	}
	VectorXd Sb(S.size());
	VectorXd Ss(S.size());
	for(int i = 0; i < S.size(); ++i)
	{
		if(fabs(S(i)) > 1)
			{ Sb(i) = S(i); Ss(i) = 1.0; }
		else
			{ Sb(i) = 1.0; Ss(i) = S(i); }
	}

	MatrixXd H = Sb.asDiagonal().inverse() * U.transpose() + Ss.asDiagonal()*V.transpose();
	MatrixXd g = H.colPivHouseholderQr().solve(Sb.asDiagonal().inverse()*U.transpose());

	if(sigma == +1) gu = g;
	else if(sigma == -1) gd = g;
	else assert(false);
}

MatrixXd Hubbard::makeBlockBl(int l0, int sigma)
{
	MatrixXd r = makeBl(l0, sigma);
	for(int i = 1; i < blockSize; ++i)
		r = makeBl((l0+i)%L, sigma)*r;
	return r;
}

MatrixXd Hubbard::makeBl(int l, int sigma)
{
	MatrixXd r = expK;
	for(int i = 0; i < N; ++i)
		r.col(i) *= exp(sigma*lambda*s(i,l));
	return r;
}

/** do one sweep of the simulation */
void Hubbard::thermalize()
{
	// start with fresh greens functions
	computeGreens(0, +1);
	computeGreens(0, -1);

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

		if(l%5 == 0)
		{
			computeGreens(l+1, +1);
			computeGreens(l+1, -1);
		}
		else
		{
			// 'wrap' the greens functions
			MatrixXd Blu = makeBl(l,+1);
			gu = Blu * gu * Blu.inverse();
			MatrixXd Bld = makeBl(l,-1);
			gd = Bld * gd * Bld.inverse();
		}

	}

	// end with fresh greens functions
	computeGreens(0, +1);
	computeGreens(0, -1);
}

void Hubbard::measure()
{
	for(int i = 0; i < N; ++i)
	{
		nn += 1.0;
		nu += 1.0-gu(i,i);
		nd += 1.0-gd(i,i);
		nud += (1.0-gu(i,i))*(1.0-gd(i,i));
	}
}

void Hubbard::clearStats()
{
	nn = 0;
	nu = 0;
	nd = 0;
	nud = 0;
}

void Hubbard::print() const
{
	std::cout << "<up> = " << nu/nn << std::endl;
	std::cout << "<down> = " << nu/nn << std::endl;
	std::cout << "<(up-down)^2> = " << mag() << std::endl;
}

double Hubbard::mag() const
{
	return (nu/nn) + (nd/nn) - 2*nud/nn;
}
