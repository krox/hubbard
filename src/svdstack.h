#include <iostream>
#include <cmath>
#include <cassert>
#include <random>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
using Eigen::MatrixXd;
using Eigen::VectorXd;


/** SVD of a matrix product: U * D * V^T = ... A_2 * A_1 * A_0 */
struct SvdStack
{
	typedef Eigen::JacobiSVD<MatrixXd,Eigen::NoQRPreconditioner> SVD;
	std::vector<SVD> stack;
	int n;
	MatrixXd tmp, V;
	int len = 0;


	SvdStack(int n, int l)
		: n(n), tmp(n,n), V(n,n)
	{
		stack.reserve(l);
		for(int i = 0; i < l; ++i)
			stack.emplace_back(n, n, Eigen::ComputeFullU | Eigen::ComputeFullV);
	}

	bool empty() const
	{
		return len == 0;
	}

	/** prepends a matrix to the decomposition */
	void push(const MatrixXd& m)
	{
		assert(m.rows() == n && m.cols() == n);
		assert(len < (int)stack.size());

		if(len == 0)
		{
			stack[len].compute(m);
			V = stack[len].matrixV();
		}
		else
		{
			tmp.noalias() = ( m * matrixU() ) * singularValues().asDiagonal();
			stack[len].compute(tmp);
			V = V * stack[len].matrixV();
		}

		len += 1;
	}

	const VectorXd& singularValues() const
	{
		assert(len > 0);
		return stack[len-1].singularValues();
	}

	const MatrixXd& matrixU() const
	{
		assert(len > 0);
		return stack[len-1].matrixU();
	}

	/*const MatrixXd& matrixV() const
	{
		assert(len > 0);
		return V; TODO
	}*/
	MatrixXd matrixV() const
	{
		assert(len > 0);
		MatrixXd r = stack[0].matrixV();
		for(int i = 1; i < len; ++i)
			r = r * stack[i].matrixV();
		return r;
	}

	void pop()
	{
		assert(len > 0);
		len -= 1;
		if(len > 0)
			V = V * stack[len].matrixV().transpose();
	}
};
