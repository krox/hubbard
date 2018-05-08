#include "util/gnuplot.h"
#include <fmt/format.h>
#include <fstream>

int Gnuplot::nplotsGlobal = 0;

Gnuplot::Gnuplot(bool persist) : plotID(nplotsGlobal++)
{
	if (persist)
		pipe = popen("gnuplot -p", "w");
	else
		pipe = popen("gnuplot", "w");
	if (pipe == nullptr)
		throw "ERROR: Could not open gnuplot (is it installed?)";

	cmd("set output\n");
	cmd("set terminal x11\n");
}

void Gnuplot::cmd(const std::string &s)
{
	assert(s.back() == '\n');
	fputs(s.c_str(), pipe);
	fflush(pipe);
}

void Gnuplot::cmd(const boost::basic_format<char> &fmt) { cmd(fmt.str()); }

void Gnuplot::plotFunction(const std::string &fun, const std::string &title)
{
	cmd(format("%s %s title \"%s\"\n") % (nplots ? "replot" : "plot") % fun %
	    (title.size() ? title : fun));
	++nplots;
}

void Gnuplot::plotData(const std::vector<double> &xs,
                       const std::vector<double> &ys, const std::string &title)
{
	std::string filename =
	    (format("gnuplot_%s_%s.txt") % plotID % nplots).str();
	std::ofstream file(filename);
	assert(xs.size() == ys.size());
	for (size_t i = 0; i < xs.size(); ++i)
		file << xs[i] << " " << ys[i] << "\n";
	file.flush();
	file.close();
	cmd(format("%s '%s' using 1:2 with %s title \"%s\"\n") %
	    (nplots ? "replot" : "plot") % filename % style % title);
	++nplots;
}

void Gnuplot::plotHistogram(const histogram &hist, const std::string &title)
{
	std::string filename =
	    (format("gnuplot_%s_%s.txt") % plotID % nplots).str();
	std::ofstream file(filename);

	for (size_t i = 0; i < hist.bins.size(); ++i)
		file << 0.5 * (hist.mins[i] + hist.maxs[i]) << " " << hist.bins[i]
		     << "\n";
	file.flush();
	file.close();
	cmd(format("%s '%s' using 1:2 with points title \"%s\"\n") %
	    (nplots ? "replot" : "plot") % filename % title);
	nplots++;
}

void Gnuplot::setLabelX(const std::string& title)
{
	cmd(format("set xlabel \"%s\"\nshow xlabel\n") % title);
}

void Gnuplot::setLabelY(const std::string& title)
{
	cmd(format("set ylabel \"%s\"\nshow ylabel\n") % title);
}

void Gnuplot::setLabelZ(const std::string& title)
{
	cmd(format("set zlabel \"%s\"\nshow zlabel\n") % title);
}

void Gnuplot::setRangeX(double min, double max)
{
	cmd(format("set xrange[%s : %s]\n") % min % max);
}

void Gnuplot::setRangeY(double min, double max)
{
	cmd(format("set yrange[%s : %s]\n") % min % max);
}

void Gnuplot::setRangeZ(double min, double max)
{
	cmd(format("set zrange[%s : %s]\n") % min % max);
}

void Gnuplot::setLogScaleX() { cmd("set logscale x\n"); }
void Gnuplot::setLogScaleY() { cmd("set logscale y\n"); }
void Gnuplot::setLogScaleZ() { cmd("set logscale z\n"); }

void Gnuplot::clear()
{
	cmd("clear\n");
	nplots = 0;
}

void Gnuplot::writePNG(const std::string& filename)
{
	cmd(fmt::format("set term png\nset output \"{}\"\nreplot\nset term x11\n", filename));
}
