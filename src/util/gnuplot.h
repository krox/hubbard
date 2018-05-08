#ifndef MESH_GNUPLOT_H
#define MESH_GNUPLOT_H

#include "boost/format.hpp"
#include "util/stats.h"

class Gnuplot
{
	FILE *pipe = nullptr;
	static int nplotsGlobal;
	int nplots = 0;
	const int plotID;
	using format = boost::format;

  public:
	std::string style = "points";

	/** constructor */
	explicit Gnuplot(bool persist = true);

	/** send a command to gnuplot */
	void cmd(const std::string &s);
	void cmd(const boost::basic_format<char> &fmt);

	/** plot a function given by a string that gnuplot can understand */
	void plotFunction(const std::string &fun, const std::string &title = "");

	/** plot a function given as Functor */
	template <typename F>
	void plotFunction(F f, double min, double max,
	                  const std::string &title = "");

	/** plot raw data points (xs[i], ys[i]) */
	void plotData(const std::vector<double> &xs, const std::vector<double> &ys,
	              const std::string &title = "data");

	/** plot a histogram */
	void plotHistogram(const histogram &hist,
	                   const std::string &title = "hist");

	/** set axis label */
	void setLabelX(const std::string& title);
	void setLabelY(const std::string& title);
	void setLabelZ(const std::string& title);

	/** set range of plot */
	void setRangeX(double min, double max);
	void setRangeY(double min, double max);
	void setRangeZ(double min, double max);

	/** make the plot logarithmic */
	void setLogScaleX();
	void setLogScaleY();
	void setLogScaleZ();

	/** remove all plots (but keep settings) */
	void clear();

	void writePNG(const std::string& filename);
};

template <typename F>
inline void Gnuplot::plotFunction(F f, double a, double b,
                                  const std::string &title)
{
	auto oldStyle = style;
	style = "lines";
	std::vector<double> xs, ys;
	for (int i = 0; i <= 100; ++i)
	{
		xs.push_back(a + (b - a) * i / 100);
		ys.push_back(f(xs.back()));
	}
	plotData(xs, ys, title);
	style = oldStyle;
}

#endif
